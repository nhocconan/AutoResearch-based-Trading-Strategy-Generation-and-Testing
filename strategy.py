#!/usr/bin/env python3
"""
Experiment #885: 1h Primary + 4h/1d HTF — Regime-Adaptive RSI Pullback with Session Filter

Hypothesis: After 600+ failed strategies, 1h timeframe with STRICT confluence filters
should generate 30-60 trades/year while maintaining positive Sharpe on ALL symbols.

Key insights from research:
1. 1h Primary TF: Target 30-60 trades/year (use VERY strict entry conditions)
2. 4h HMA(21) for primary trend direction (HTF signal)
3. 1d HMA(21) for macro regime filter (bull/bear market)
4. 1h RSI(14) pullback within HTF trend (entry timing)
5. Choppiness Index(14) regime detection: CHOP>55=range, CHOP<45=trend
6. Session filter: Only trade 8-20 UTC (highest volume, lowest whipsaw)
7. Volume confirmation: volume > 0.8x 20-bar average
8. ATR(14) trailing stop (2.5x) for risk management

Why this should work on 1h:
- HTF (4h/1d) provides strong trend bias, 1h only for entry timing
- Session filter eliminates Asian session whipsaws (low volume)
- Volume filter confirms institutional participation
- Regime-adaptive: mean-revert in range, trend-follow in trend
- Smaller position size (0.25) for lower TF fee management

Critical improvements from failed experiments:
- RELAXED RSI thresholds (30/70 not 20/80) to ensure trades on all symbols
- Session filter reduces false signals during low-volume hours
- Volume confirmation prevents entries on thin liquidity
- ALL symbols MUST have positive Sharpe (no SOL-only bias)
- Use discrete signal sizes (0.0, ±0.20, ±0.25) minimize fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 30-60 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_pullback_4h1d_hma_session_vol_chop_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 55 = ranging, CHOP < 45 = trending.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=14)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    sma_50_1h = calculate_sma(close, 50)
    sma_200_1h = calculate_sma(close, 200)
    
    # Volume MA for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align 4h HMA for primary trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro regime (bull/bear market)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(chop_1h[i]):
            continue
        if np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_50_1h[i]) or np.isnan(sma_200_1h[i]):
            continue
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_ma_20[i]
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === SHORT-TERM TREND FILTER (1h SMA50/200) ===
        above_sma50 = close[i] > sma_50_1h[i]
        below_sma50 = close[i] < sma_50_1h[i]
        above_sma200 = close[i] > sma_200_1h[i]
        below_sma200 = close[i] < sma_200_1h[i]
        
        # === REGIME DETECTION (1h Choppiness Index) ===
        ranging_regime = chop_1h[i] > 55
        trending_regime = chop_1h[i] < 45
        
        # === RSI SIGNALS (Relaxed thresholds: 30/70 for more trades) ===
        rsi_oversold = rsi_1h[i] < 35
        rsi_overbought = rsi_1h[i] > 65
        rsi_extreme_oversold = rsi_1h[i] < 25
        rsi_extreme_overbought = rsi_1h[i] > 75
        rsi_neutral = 40 <= rsi_1h[i] <= 60
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 55) — Mean Reversion ===
        if ranging_regime and in_session and volume_confirmed:
            # Long: RSI oversold + macro/primary trend not strongly bearish
            if rsi_oversold and not (macro_bear and trend_4h_bearish):
                desired_signal = BASE_SIZE
            
            # Short: RSI overbought + macro/primary trend not strongly bullish
            if rsi_overbought and not (macro_bull and trend_4h_bullish):
                desired_signal = -BASE_SIZE
            
            # Fallback: extreme RSI alone (guarantees trades)
            if rsi_extreme_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if rsi_extreme_overbought and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 45) — Trend Following ===
        elif trending_regime and in_session and volume_confirmed:
            # Long: Bullish trend + RSI pullback (not overbought)
            if (macro_bull or trend_4h_bullish) and rsi_oversold:
                desired_signal = BASE_SIZE
            elif (macro_bull or trend_4h_bullish) and rsi_neutral and above_sma50:
                desired_signal = REDUCED_SIZE
            
            # Short: Bearish trend + RSI bounce (not oversold)
            if (macro_bear or trend_4h_bearish) and rsi_overbought:
                desired_signal = -BASE_SIZE
            elif (macro_bear or trend_4h_bearish) and rsi_neutral and below_sma50:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: RSI extremes with trend confluence
            if in_session and volume_confirmed:
                if rsi_extreme_oversold and (macro_bull or trend_4h_bullish or above_sma200):
                    desired_signal = REDUCED_SIZE
                
                if rsi_extreme_overbought and (macro_bear or trend_4h_bearish or below_sma200):
                    desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and RSI not overbought
                if (macro_bull or trend_4h_bullish) and rsi_1h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and RSI not oversold
                if (macro_bear or trend_4h_bearish) and rsi_1h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro + primary trend reverses + RSI overbought
            if macro_bear and trend_4h_bearish and rsi_1h[i] > 70:
                desired_signal = 0.0
            # Exit if RSI extremely overbought in ranging regime
            if ranging_regime and rsi_1h[i] > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro + primary trend reverses + RSI oversold
            if macro_bull and trend_4h_bullish and rsi_1h[i] < 30:
                desired_signal = 0.0
            # Exit if RSI extremely oversold in ranging regime
            if ranging_regime and rsi_1h[i] < 25:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
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