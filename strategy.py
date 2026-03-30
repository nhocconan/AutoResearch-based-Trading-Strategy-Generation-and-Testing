#!/usr/bin/env python3
"""
Experiment #002: 12h Bollinger Band Squeeze + 1d Trend + Volume Confirmation

HYPOTHESIS: BB Width compression signals impending volatility expansion.
When BB Width contracts to 20d lows, price typically breaks out with momentum.
Combined with 1d SMA200 for trend direction and volume confirmation,
this captures the "volatility squeeze breakout" pattern that works in both
bull (upward breakouts) and bear (downward breakouts) markets.

WHY 12h: Between 4h (too many trades) and 1d (too few trades).
BB squeeze signals are rare enough on 12h to hit the 50-150 trade target.
Squeeze patterns are more reliable than pure price channels because they
explicitly measure volatility contraction before expansion.

KEY INSIGHT FROM DB: "BB Width at 30d low" is explicitly mentioned as a 
winning pattern. This strategy implements that exact concept.

Entry: BB Width < 20th percentile (30d) + BB% > 0.8 + close > SMA200 + vol > 1.5x MA
Exit: 2.0 ATR stoploss, RSI extremes for early exit
Position: 0.30 (moderate)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_bb_squeeze_1d_trend_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, num_std=2):
    """Bollinger Bands + Width + %B"""
    n = len(close)
    mid = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = mid + num_std * std
    lower = mid - num_std * std
    
    # BB Width (normalized by mid band)
    bb_width = np.full(n, np.nan)
    for i in range(period, n):
        if mid[i] > 0:
            bb_width[i] = (upper[i] - lower[i]) / mid[i]
    
    # BB %B
    bb_pctb = np.full(n, np.nan)
    for i in range(period, n):
        if upper[i] - lower[i] > 0:
            bb_pctb[i] = (close[i] - lower[i]) / (upper[i] - lower[i])
    
    return upper, mid, lower, bb_width, bb_pctb

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend direction
    sma_200_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # Local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    bb_upper, bb_mid, bb_lower, bb_width, bb_pctb = calculate_bollinger_bands(close, period=20, num_std=2)
    
    # BB Width percentile (30d lookback) - vectorized
    bb_width_pct = np.full(n, np.nan)
    lookback = 30
    for i in range(lookback, n):
        window_vals = bb_width[i-lookback+1:i+1]
        valid = window_vals[~np.isnan(window_vals)]
        if len(valid) >= lookback * 0.7:
            min_val = np.min(valid)
            max_val = np.max(valid)
            if max_val > min_val:
                bb_width_pct[i] = (bb_width[i] - min_val) / (max_val - min_val)
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI for exit filter
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    signals = np.zeros(n)
    SIZE = 0.30  # Moderate sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_bar = 0
    
    warmup = 250  # Need 200 for SMA200 + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_200_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_width_pct[i]) or np.isnan(bb_pctb[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1d TREND (SMA200) ===
        trend_up = close[i] > sma_200_aligned[i]
        
        # === BB SQUEEZE CONDITIONS ===
        # BB Width in bottom 20% of 30d range = squeeze
        is_squeezed = bb_width_pct[i] < 0.20
        
        # Price at upper BB = breakout momentum
        breakout_up = bb_pctb[i] > 0.80
        
        # Price at lower BB = bearish breakout
        breakout_down = bb_pctb[i] < 0.20
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === MINIMUM VOLATILITY FILTER ===
        # ATR must be > 0.5% of price (avoid trading in low vol)
        min_atr_ratio = 0.005
        if atr_14[i] / close[i] < min_atr_ratio:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # Track bars in position
        if in_position:
            bars_held = i - entry_bar
        else:
            bars_held = 0
        
        if not in_position:
            # === LONG: BB squeeze + breakout up + trend up + volume spike ===
            if is_squeezed and breakout_up and trend_up and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: BB squeeze + breakout down + trend down + volume spike ===
            # (trend_down = not trend_up, which means price < SMA200)
            if is_squeezed and breakout_down and not trend_up and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.0 ATR) ===
        if in_position:
            atr_entry = atr_14[i]  # Use current ATR for stop calculation
            
            if position_side > 0:
                stop_price = close[entry_bar] - 2.0 * atr_entry
                if low[i] < stop_price:
                    desired_signal = 0.0
            
            if position_side < 0:
                stop_price = close[entry_bar] + 2.0 * atr_entry
                if high[i] > stop_price:
                    desired_signal = 0.0
        
        # === RSI EXIT FILTER ===
        if in_position and bars_held >= 2:
            if position_side > 0 and rsi[i] > 75:
                desired_signal = 0.0
            if position_side < 0 and rsi[i] < 25:
                desired_signal = 0.0
        
        # === TRAILING STOP (activate after 4 bars) ===
        if in_position and bars_held >= 4:
            if position_side > 0:
                # Trail stop: highest since entry - 2.5 ATR
                highest = np.max(high[entry_bar:i+1])
                trailing_stop = highest - 2.5 * atr_14[i]
                stop_price = trailing_stop
                if low[i] < stop_price:
                    desired_signal = 0.0
            
            if position_side < 0:
                # Trail stop: lowest since entry + 2.5 ATR
                lowest = np.min(low[entry_bar:i+1])
                trailing_stop = lowest + 2.5 * atr_14[i]
                stop_price = trailing_stop
                if high[i] > stop_price:
                    desired_signal = 0.0
        
        # === TIME EXIT (max 12 bars ~ 6 days) ===
        if bars_held >= 12:
            desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_bar = i
            else:
                # Same direction - maintain
                pass
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals