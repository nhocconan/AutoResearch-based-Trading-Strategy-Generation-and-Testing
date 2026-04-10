#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot long/short with 4h trend filter and volume confirmation
# - Long when price > H3 (bullish bias) AND 4h close > 4h EMA20 AND volume > 1.3x 20-bar avg
# - Short when price < L3 (bearish bias) AND 4h close < 4h EMA20 AND volume > 1.3x 20-bar avg
# - Exit when price touches H4/L4 or opposite pivot level
# - Uses discrete position sizing (0.20) to minimize fee churn
# - Targets ~25 trades/year (100 total over 4 years) to avoid fee drag
# - Session filter: 08-20 UTC to avoid low-liquidity hours
# - Camarilla pivots provide intraday support/resistance levels that work in ranging markets
# - 4h EMA20 ensures we trade with higher timeframe momentum
# - Volume confirmation ensures breakouts have institutional participation

name = "1h_camarilla_pivot_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-calculate session hours for efficiency
    hours = prices.index.hour
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h EMA(20) for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Pre-compute 1h volume confirmation: > 1.3x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.3 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(volume_20_avg[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla pivots for today (based on previous day's range)
        # Need daily high/low/close from prior day
        if i < 24:  # Need at least 24 hours of data for 1-day lookback
            signals[i] = 0.0
            continue
            
        # Get previous day's OHLC (24 hours ago for 1h timeframe)
        prev_high = prices['high'].iloc[i-24]
        prev_low = prices['low'].iloc[i-24]
        prev_close = prices['close'].iloc[i-24]
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        # Camarilla pivot levels
        h3 = prev_close + range_val * 1.1 / 4
        l3 = prev_close - range_val * 1.1 / 4
        h4 = prev_close + range_val * 1.1 / 2
        l4 = prev_close - range_val * 1.1 / 2
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price > H3 (bullish bias) AND 4h uptrend AND volume spike
            if (prices['close'].iloc[i] > h3 and 
                prices['close'].iloc[i] > ema_20_4h_aligned[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.20
            # Short signal: price < L3 (bearish bias) AND 4h downtrend AND volume spike
            elif (prices['close'].iloc[i] < l3 and 
                  prices['close'].iloc[i] < ema_20_4h_aligned[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.20
        else:  # Have position - look for exit
            # Exit conditions
            exit_signal = False
            if position == 1:  # Long position
                # Exit if price touches H4 (strong resistance) or returns to L3 (invalidates bias)
                if prices['high'].iloc[i] >= h4 or prices['close'].iloc[i] <= l3:
                    exit_signal = True
            else:  # Short position
                # Exit if price touches L4 (strong support) or returns to H3 (invalidates bias)
                if prices['low'].iloc[i] <= l4 or prices['close'].iloc[i] >= h3:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals