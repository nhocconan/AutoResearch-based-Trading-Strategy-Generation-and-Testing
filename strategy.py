#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level in 4h uptrend (EMA21 rising) with volume spike
# - Short when price breaks below Camarilla L3 level in 4h downtrend (EMA21 falling) with volume spike
# - Uses discrete position sizing (0.20) to minimize fee churn
# - Targets 15-37 trades/year (60-150 total over 4 years) to avoid fee drag
# - 4h trend filter reduces false breakouts in ranging markets
# - Session filter (08-20 UTC) to reduce noise trades
# - ATR-based stoploss to limit drawdown

name = "1h_4h_camarilla_breakout_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h indicators
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # 4h EMA(21) for trend filter and slope
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_slope_4h = np.diff(ema_21_4h, prepend=ema_21_4h[0])
    
    # 4h volume confirmation: > 1.5x 20-period average
    avg_volume_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike_4h = volume_4h > (1.5 * avg_volume_20_4h)
    
    # Align HTF indicators to 1h timeframe
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    ema_slope_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_slope_4h)
    vol_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_spike_4h)
    
    # Pre-compute ATR for stoploss (using 1h data)
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift(1))
    low_close = np.abs(prices['low'] - prices['close'].shift(1))
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_ranges = np.nanmax(ranges.values, axis=1)
    atr_14 = pd.Series(true_ranges).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop_multiplier = 2.5
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(ema_slope_4h_aligned[i]) or 
            np.isnan(vol_spike_4h_aligned[i]) or np.isnan(atr_14[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # ATR-based stoploss
            if prices['close'].iloc[i] < entry_price - atr_stop_multiplier * atr_14[i]:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # ATR-based stoploss
            if prices['close'].iloc[i] > entry_price + atr_stop_multiplier * atr_14[i]:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Calculate Camarilla pivot levels on previous completed 4h bar
            # Get index of completed 4h bar (each 4h bar = 4 * 1h bars)
            completed_4h_bars = i // 4
            if completed_4h_bars >= 1 and completed_4h_bars < len(df_4h):
                # Use previous completed 4h bar for Camarilla calculation
                prev_4h_idx = completed_4h_bars - 1
                if prev_4h_idx >= 0:
                    high = df_4h['high'].iloc[prev_4h_idx]
                    low = df_4h['low'].iloc[prev_4h_idx]
                    close = df_4h['close'].iloc[prev_4h_idx]
                    
                    # Calculate Camarilla levels
                    range_val = high - low
                    camarilla_h3 = close + range_val * 1.1 / 4
                    camarilla_l3 = close - range_val * 1.1 / 4
                    
                    # Long signal: price breaks above Camarilla H3 in 4h uptrend with volume spike
                    if (prices['high'].iloc[i] > camarilla_h3 and 
                        ema_slope_4h_aligned[i] > 0 and  # EMA21 rising = uptrend
                        vol_spike_4h_aligned[i]):
                        position = 1
                        entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                        signals[i] = 0.20
                    # Short signal: price breaks below Camarilla L3 in 4h downtrend with volume spike
                    elif (prices['low'].iloc[i] < camarilla_l3 and 
                          ema_slope_4h_aligned[i] < 0 and  # EMA21 falling = downtrend
                          vol_spike_4h_aligned[i]):
                        position = -1
                        entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                        signals[i] = -0.20
    
    return signals