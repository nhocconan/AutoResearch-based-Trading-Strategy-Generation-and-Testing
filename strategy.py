#!/usr/bin/env python3
"""
Experiment #248: 30m Primary + 4h/1d HTF — Trend-Following Mean Reversion

Hypothesis: After repeated failures with complex regime-switching (#238, #240, #242 = 0 trades),
simplify to proven pattern: HTF trend direction + LTF mean reversion entries.

Key changes from failed 30m attempts:
1. 4h HMA(21) for MACRO TREND DIRECTION (only trade with HTF trend)
2. 30m RSI(7) + Z-score(20) for ENTRY TIMING (pullback within trend)
3. Volume confirmation (>1.2x 20-bar avg) to filter low-liquidity traps
4. Session filter (8-20 UTC) to avoid Asian session whipsaws
5. ATR(14) 2.0x trailing stoploss for risk management

TARGET: 40-80 trades/year on 30m, Sharpe > 0.5 on ALL symbols (BTC/ETH/SOL)
CRITICAL: Entry conditions loose enough for trades but strict enough for quality.
Position size: 0.25 (conservative for 30m volatility)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_trend_rsi_zscore_vol_session_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
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

def calculate_zscore(close, period=20):
    """Calculate Z-score of price vs rolling mean."""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = (close_s - rolling_mean) / (rolling_std + 1e-10)
    return zscore.fillna(0.0).values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    volume_s = pd.Series(volume)
    vol_avg = volume_s.rolling(window=period, min_periods=period).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_ratio = volume_s / (vol_avg + 1e-10)
    return vol_ratio.fillna(1.0).values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

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
    
    # Calculate 30m indicators (primary timeframe)
    rsi_7 = calculate_rsi(close, period=7)
    zscore_20 = calculate_zscore(close, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Calculate 4h HMA for macro trend (aligned properly)
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1d HMA for broader trend (aligned properly)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.25
    POSITION_SIZE_HALF = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(rsi_7[i]) or np.isnan(zscore_20[i]):
            signals[i] = 0.0
            continue
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = (utc_hour >= 8) and (utc_hour <= 20)
        
        # === MACRO TREND (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        trend_bullish = price_above_hma_4h
        trend_bearish = price_below_hma_4h
        
        # === BROADER TREND (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        macro_bullish = price_above_hma_1d
        macro_bearish = price_below_hma_1d
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 1.2
        
        # === ENTRY SIGNALS ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + RSI oversold + Z-score low + volume + session
        if trend_bullish and in_session:
            # RSI(7) < 35 = oversold pullback in uptrend
            # Z-score < -1.0 = price below mean
            # Volume > 1.2x = confirmation
            if rsi_7[i] < 35.0 and zscore_20[i] < -1.0:
                if volume_confirmed or macro_bullish:
                    desired_signal = POSITION_SIZE
                else:
                    desired_signal = POSITION_SIZE_HALF
        
        # SHORT: 4h bearish + RSI overbought + Z-score high + volume + session
        elif trend_bearish and in_session:
            # RSI(7) > 65 = overbought rally in downtrend
            # Z-score > 1.0 = price above mean
            # Volume > 1.2x = confirmation
            if rsi_7[i] > 65.0 and zscore_20[i] > 1.0:
                if volume_confirmed or macro_bearish:
                    desired_signal = -POSITION_SIZE
                else:
                    desired_signal = -POSITION_SIZE_HALF
        
        # === STOPLOSS CHECK (2.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 4h trend flips bearish
        if in_position and position_side > 0 and trend_bearish:
            desired_signal = 0.0
        
        # Exit short if 4h trend flips bullish
        if in_position and position_side < 0 and trend_bullish:
            desired_signal = 0.0
        
        # === RSI EXTREME EXIT (mean reversion complete) ===
        # Exit long if RSI goes overbought
        if in_position and position_side > 0 and rsi_7[i] > 70.0:
            desired_signal = 0.0
        
        # Exit short if RSI goes oversold
        if in_position and position_side < 0 and rsi_7[i] < 30.0:
            desired_signal = 0.0
        
        # === Z-SCORE EXIT (back to mean) ===
        # Exit long if Z-score > 0.5 (back above mean)
        if in_position and position_side > 0 and zscore_20[i] > 0.5:
            desired_signal = 0.0
        
        # Exit short if Z-score < -0.5 (back below mean)
        if in_position and position_side < 0 and zscore_20[i] < -0.5:
            desired_signal = 0.0
        
        # === HOLD LOGIC - maintain position if setup still valid ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if RSI not overbought AND trend still bullish
                if rsi_7[i] < 70.0 and trend_bullish:
                    desired_signal = POSITION_SIZE_HALF
            elif position_side < 0:
                # Hold short if RSI not oversold AND trend still bearish
                if rsi_7[i] > 30.0 and trend_bearish:
                    desired_signal = -POSITION_SIZE_HALF
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                if position_side > 0:
                    highest_since_entry = close[i]
                    lowest_since_entry = float('inf')
                else:
                    lowest_since_entry = close[i]
                    highest_since_entry = 0.0
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                if position_side > 0:
                    highest_since_entry = close[i]
                    lowest_since_entry = float('inf')
                else:
                    lowest_since_entry = close[i]
                    highest_since_entry = 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals