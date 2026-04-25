#!/usr/bin/env python3
"""
1h Camarilla Pivot Reversal with 4h EMA Trend Filter and Volume Spike
Hypothesis: Camarilla H3/L3 levels act as intraday support/resistance. Reversals from these levels with 4h EMA50 trend alignment capture mean-reversion moves in ranging markets and pullbacks in trending markets. Volume spike confirms participation. Works in bull markets via buying dips to L3 in uptrend, bear markets via selling rallies to H3 in downtrend. Discrete position sizing (0.20) controls drawdown. Session filter (08-20 UTC) reduces noise trades. Target: 15-35 trades/year on 1h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session hours (08-20 UTC) once before loop
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate ATR(14) for stoploss
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Pre-compute 20-period volume MA for volume spike detection
    vol_ma_20 = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for ATR(14) and EMA50 to propagate
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(atr[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50 = ema_50_4h_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Calculate Camarilla pivot levels for today (using previous day's OHLC)
        # We need to get the previous day's high, low, close
        if i >= 96:  # 96 * 1h = 96 hours = 4 days, enough to get previous day
            # Get index of 1h bars that belong to previous day (24 bars back)
            prev_day_idx = i - 24
            if prev_day_idx >= 24:  # ensure we have enough data for previous day
                # Get high, low, close of the previous day (24h period)
                day_high = np.max(high[prev_day_idx-24:prev_day_idx])
                day_low = np.min(low[prev_day_idx-24:prev_day_idx])
                day_close = close[prev_day_idx-1]  # close of previous day
                
                # Calculate Camarilla levels
                range_val = day_high - day_low
                if range_val > 0:
                    camarilla_h3 = day_close + (range_val * 1.1 / 4)
                    camarilla_l3 = day_close - (range_val * 1.1 / 4)
                else:
                    camarilla_h3 = curr_high
                    camarilla_l3 = curr_low
            else:
                camarilla_h3 = curr_high
                camarilla_l3 = curr_low
        else:
            camarilla_h3 = curr_high
            camarilla_l3 = curr_low
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long: reversal from Camarilla L3 (price > L3 and rising) AND uptrend AND volume spike
            long_condition = (curr_close > camarilla_l3 and 
                            curr_close > close[i-1] and  # price rising from L3
                            curr_close > ema_50 and      # uptrend filter
                            volume_spike)
            # Short: reversal from Camarilla H3 (price < H3 and falling) AND downtrend AND volume spike
            short_condition = (curr_close < camarilla_h3 and 
                             curr_close < close[i-1] and  # price falling from H3
                             curr_close < ema_50 and      # downtrend filter
                             volume_spike)
            
            if long_condition:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price falls below EMA50 or reversal signal
            if (curr_close <= entry_price - 2.0 * atr_val or 
                curr_close < ema_50 or
                (curr_close < camarilla_h3 and curr_close < close[i-1])):  # reversal from H3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price rises above EMA50 or reversal signal
            if (curr_close >= entry_price + 2.0 * atr_val or 
                curr_close > ema_50 or
                (curr_close > camarilla_l3 and curr_close > close[i-1])):  # reversal from L3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Reversal_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0