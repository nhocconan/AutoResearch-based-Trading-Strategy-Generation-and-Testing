#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike
# Uses 4h EMA50 for trend direction (works in bull/bear), 1h Camarilla levels for precise entry,
# volume confirmation to avoid false breakouts. Session filter (08-20 UTC) reduces noise.
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.
# Signal size: 0.20 discrete levels to balance profit and risk.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate ATR(14) for stoploss and volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR percentile for volatility regime filter (avoid high volatility chop)
    atr_percentile = pd.Series(atr).rolling(window=50, min_periods=20).apply(
        lambda x: np.percentile(x, 50) if len(x) >= 20 else np.nan, raw=True
    ).values
    vol_regime_filter = atr <= atr_percentile  # Only trade in low/medium volatility regimes
    
    # Calculate Camarilla levels from previous day (using 1d data for accuracy)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous day's OHLC
    prev_close = np.concatenate([[np.nan], df_1d['close'].values[:-1]])
    prev_high = np.concatenate([[np.nan], df_1d['high'].values[:-1]])
    prev_low = np.concatenate([[np.nan], df_1d['low'].values[:-1]])
    
    # Align previous day's data to 1h timeframe
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = prev_close_aligned + 1.0 * (prev_high_aligned - prev_low_aligned)
    camarilla_s3 = prev_close_aligned - 1.0 * (prev_high_aligned - prev_low_aligned)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Session filter: 08-20 UTC (pre-compute hours for efficiency)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    max_high_since_entry = 0.0  # For trailing stop
    min_low_since_entry = 0.0   # For trailing stop
    
    start_idx = max(50, 20, 14)  # warmup for EMA50, volume, ATR
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_open = open_price[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        curr_atr = atr[i]
        curr_r3 = camarilla_r3[i]
        curr_s3 = camarilla_s3[i]
        curr_volume_confirm = volume_confirm[i]
        curr_vol_regime = vol_regime_filter[i]
        curr_session = session_filter[i]
        
        # Handle position exits and stops
        if position == 1:  # Long position
            # Update trailing stop: highest high since entry
            max_high_since_entry = max(max_high_since_entry, curr_high)
            # Dynamic stoploss: ATR-based trailing stop
            trail_stop = max_high_since_entry - 2.5 * curr_atr
            # Fixed stoploss: 2.0 * ATR below entry
            fixed_stop = entry_price - 2.0 * atr_at_entry
            # Use the tighter of the two stops
            stop_price = max(trail_stop, fixed_stop)
            
            # Exit conditions:
            # 1. Stoploss hit (trailing or fixed)
            # 2. Price crosses below 4h EMA50 (trend change)
            # 3. Price drops below Camarilla S3 (breakout failed)
            # 4. Volatility regime shifts to high (avoid chop)
            # 5. Outside trading session
            if (curr_low <= stop_price or
                curr_close < curr_ema_50_4h or
                curr_close < curr_s3 or
                not curr_vol_regime or
                not curr_session):
                signals[i] = 0.0
                position = 0
                max_high_since_entry = 0.0
                min_low_since_entry = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Update trailing stop: lowest low since entry
            min_low_since_entry = min(min_low_since_entry, curr_low)
            # Dynamic stoploss: ATR-based trailing stop
            trail_stop = min_low_since_entry + 2.5 * curr_atr
            # Fixed stoploss: 2.0 * ATR above entry
            fixed_stop = entry_price + 2.0 * atr_at_entry
            # Use the tighter of the two stops
            stop_price = min(trail_stop, fixed_stop)
            
            # Exit conditions:
            # 1. Stoploss hit (trailing or fixed)
            # 2. Price crosses above 4h EMA50 (trend change)
            # 3. Price rises above Camarilla R3 (breakout failed)
            # 4. Volatility regime shifts to high (avoid chop)
            # 5. Outside trading session
            if (curr_high >= stop_price or
                curr_close > curr_ema_50_4h or
                curr_close > curr_r3 or
                not curr_vol_regime or
                not curr_session):
                signals[i] = 0.0
                position = 0
                max_high_since_entry = 0.0
                min_low_since_entry = 0.0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Only enter in session and low/medium volatility regimes to avoid whipsaws
            if not (curr_session and curr_vol_regime):
                signals[i] = 0.0
                continue
                
            # Long entry: price breaks above Camarilla R3 + above 4h EMA50 + volume confirm
            if (curr_close > curr_r3 and
                curr_close > curr_ema_50_4h and
                curr_volume_confirm):
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
                max_high_since_entry = curr_high
                min_low_since_entry = curr_low
            # Short entry: price breaks below Camarilla S3 + below 4h EMA50 + volume confirm
            elif (curr_close < curr_s3 and
                  curr_close < curr_ema_50_4h and
                  curr_volume_confirm):
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
                max_high_since_entry = curr_high
                min_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals