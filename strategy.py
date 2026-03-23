#!/usr/bin/env python3
"""
Experiment #348: 30m Primary + 4h/1d HTF — Regime-Adaptive Pullback Strategy

Hypothesis: Previous 30m/1h strategies failed because:
1. Used LTF (30m/1h) for regime detection → too noisy, frequent whipsaw
2. Symmetric long/short logic → destroyed in 2022 crash and 2025 bear
3. Too many trades (>200/year) → fee drag kills all profits
4. Volume filters too strict → 0 trades on some symbols

This strategy uses HTF for all major decisions:
1. 1d Choppiness Index for REGIME (CHOP>55=range, CHOP<45=trend) — MORE STABLE than LTF
2. 4h HMA(21) for TREND BIAS (hard filter: only long if 4h bullish, only short if 4h bearish)
3. 30m RSI(7) for ENTRY TIMING (pullback entries within HTF trend)
4. Session filter: only 8-20 UTC (highest volume, lowest whipsaw)
5. Asymmetric sizing: 0.25 in trend regime, 0.15 in range regime

KEY INSIGHT: Use HTF (1d/4h) for DIRECTION and REGIME, use 30m only for ENTRY TIMING.
This gives HTF trade frequency (~40-60/year) with 30m execution precision.
Session filter eliminates Asian session whipsaw (0-8 UTC).

TARGET: 40-60 trades/year on 30m, Sharpe > 0.5 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_pullback_4h_trend_1d_regime_session_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2 * wma_half - wma_full
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return pd.to_datetime(open_time, unit='ms').dt.hour.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 30m indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)  # Faster RSI for entry timing
    rsi_14 = calculate_rsi(close, period=14)
    sma_200 = calculate_sma(close, period=200)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d Choppiness for regime detection
    chop_1d_raw = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
    # Extract UTC hour for session filter
    utc_hour = get_utc_hour(open_time)
    
    signals = np.zeros(n)
    BASE_SIZE_TREND = 0.25  # 25% position size in trend regime
    BASE_SIZE_RANGE = 0.15  # 15% position size in range regime (lower conviction)
    
    # Position tracking for stoploss/takeprofit
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(sma_200[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(chop_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        # Eliminates Asian session whipsaw (0-8 UTC) and late night low volume
        in_session = (utc_hour[i] >= 8) and (utc_hour[i] <= 20)
        
        if not in_session:
            # Outside session: maintain position if already in, but don't enter new
            if in_position:
                signals[i] = signals[i-1] if i > 0 else 0.0
            else:
                signals[i] = 0.0
            continue
        
        # === MACRO BIAS (4h HMA - HARD FILTER) ===
        # Only take LONGS if price above 4h HMA (bullish trend)
        # Only take SHORTS if price below 4h HMA (bearish trend)
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (1d Choppiness - MORE STABLE) ===
        is_choppy = chop_1d_aligned[i] > 55.0  # High choppiness = range regime (mean revert)
        is_trending = chop_1d_aligned[i] < 45.0  # Low choppiness = trend regime (pullback)
        
        # === VOLUME CONFIRMATION (relaxed for trade generation) ===
        volume_confirmed = volume[i] > 0.8 * vol_ma_20[i]  # 80% of avg is enough
        
        # === SMA200 FILTER (additional trend confirmation) ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_trending:
            # TREND REGIME: RSI pullback entries aligned with 4h trend
            # Long: 4h bullish + RSI(7) pullback to 35-45 + volume confirmed
            # Short: 4h bearish + RSI(7) rally to 55-65 + volume confirmed
            
            rsi_pullback_long = (rsi_7[i] >= 30) and (rsi_7[i] <= 45)
            rsi_rally_short = (rsi_7[i] >= 55) and (rsi_7[i] <= 70)
            
            # Long: 4h HMA + SMA200 bullish + RSI pullback + volume
            if price_above_hma_4h and price_above_sma200 and rsi_pullback_long and volume_confirmed:
                desired_signal = BASE_SIZE_TREND
            
            # Short: 4h HMA + SMA200 bearish + RSI rally + volume
            elif price_below_hma_4h and price_below_sma200 and rsi_rally_short and volume_confirmed:
                desired_signal = -BASE_SIZE_TREND
        
        elif is_choppy:
            # RANGE REGIME: Mean reversion at extremes (smaller size)
            # Long: 4h bullish + RSI(7) < 25 (oversold)
            # Short: 4h bearish + RSI(7) > 75 (overbought)
            
            if price_above_hma_4h and rsi_7[i] < 25:
                desired_signal = BASE_SIZE_RANGE
            
            elif price_below_hma_4h and rsi_7[i] > 75:
                desired_signal = -BASE_SIZE_RANGE
        
        else:
            # NEUTRAL REGIME (45 <= CHOP <= 55): Wait for clarity
            # Only take extreme RSI signals with strong trend alignment
            if price_above_hma_4h and price_above_sma200 and rsi_7[i] < 20:
                desired_signal = BASE_SIZE_RANGE * 0.8
            
            elif price_below_hma_4h and price_below_sma200 and rsi_7[i] > 80:
                desired_signal = -BASE_SIZE_RANGE * 0.8
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT (2R target for lower TF) ===
        take_profit_triggered = False
        
        if in_position and position_side > 0:
            profit = close[i] - entry_price
            if profit >= 2.0 * entry_atr:
                take_profit_triggered = True
        
        if in_position and position_side < 0:
            profit = entry_price - close[i]
            if profit >= 2.0 * entry_atr:
                take_profit_triggered = True
        
        if take_profit_triggered:
            desired_signal = 0.0
        
        # === RSI EXTREME EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and rsi_7[i] > 70:
            # Long position: exit when RSI reaches overbought
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_7[i] < 30:
            # Short position: exit when RSI reaches oversold
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered and not take_profit_triggered:
            # Check if trend and regime still valid
            if position_side > 0:
                if price_above_hma_4h and (is_trending or (is_choppy and rsi_7[i] < 70)):
                    desired_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                if price_below_hma_4h and (is_trending or (is_choppy and rsi_7[i] > 30)):
                    desired_signal = signals[i-1] if i > 0 else 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals