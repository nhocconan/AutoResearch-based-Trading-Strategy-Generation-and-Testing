#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter, volume confirmation, and session filter
# - Long when price breaks above H3 Camarilla level in 4h uptrend (close > EMA50) with volume > 1.3x 20-bar avg during 08-20 UTC
# - Short when price breaks below L3 Camarilla level in 4h downtrend (close < EMA50) with volume spike during 08-20 UTC
# - Uses discrete position sizing (0.20) to minimize fee churn
# - Targets ~25 trades/year (100 total over 4 years) to avoid fee drag
# - 4h trend filter reduces false breakouts in ranging markets
# - Session filter (08-20 UTC) avoids low-liquidity periods
# - Camarilla pivots provide structured support/resistance levels effective in both bull and bear markets

name = "1h_4h_camarilla_breakout_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session hours (08-20 UTC) once before loop
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h indicators
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 4h volume confirmation: > 1.3x 20-period average
    avg_volume_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike_4h = volume_4h > (1.3 * avg_volume_20_4h)
    vol_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_spike_4h)
    
    # Pre-compute ATR for stoploss (using 1h data)
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift(1))
    low_close = np.abs(prices['low'] - prices['close'].shift(1))
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_ranges = np.nanmax(ranges.values, axis=1)
    atr_14 = pd.Series(true_ranges).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop_multiplier = 2.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_spike_4h_aligned[i]) or 
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
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
            # Calculate Camarilla levels on 1h data (using previous completed bar)
            if i >= 20:
                # Use lookback of 20 completed bars (excluding current)
                lookback_start = i - 20
                lookback_end = i  # exclusive
                
                high_prev = prices['high'].iloc[lookback_start:lookback_end].max()
                low_prev = prices['low'].iloc[lookback_start:lookback_end].min()
                close_prev = prices['close'].iloc[lookback_end - 1]  # Previous bar close
                
                # Camarilla levels: H3/L3 are key breakout levels
                range_prev = high_prev - low_prev
                h3 = close_prev + range_prev * 1.1 / 4
                l3 = close_prev - range_prev * 1.1 / 4
                
                # Long signal: price breaks above H3 in 4h uptrend with volume spike
                if (prices['close'].iloc[i] > h3 and 
                    prices['close'].iloc[i] > ema_50_4h_aligned[i] and 
                    vol_spike_4h_aligned[i]):
                    position = 1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    signals[i] = 0.20
                # Short signal: price breaks below L3 in 4h downtrend with volume spike
                elif (prices['close'].iloc[i] < l3 and 
                      prices['close'].iloc[i] < ema_50_4h_aligned[i] and 
                      vol_spike_4h_aligned[i]):
                    position = -1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    signals[i] = -0.20
    
    return signals