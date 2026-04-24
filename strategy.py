#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA34 trend filter and 1d volume spike confirmation.
- Long when price breaks above Camarilla H3 AND 4h EMA34 > prior 4h EMA34 (uptrend) AND 1d volume > 1.5x 20d average volume
- Short when price breaks below Camarilla L3 AND 4h EMA34 < prior 4h EMA34 (downtrend) AND 1d volume > 1.5x 20d average volume
- Exit on opposite Camarilla break (L3 for long, H3 for short) or 4h EMA34 trend reversal
- Uses 1h primary with 4h/1d HTF to target 15-37 trades/year (60-150 over 4 years)
- Camarilla levels provide intraday support/resistance; EMA34 filters chop; volume spike confirms institutional participation
- Fixed size 0.20 to control fees and drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels (H3, L3) from prior day
    # Typical Camarilla: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    # We use prior daily OHLC to avoid look-ahead
    daily_ohlc = get_htf_data(prices, '1d')
    if len(daily_ohlc) < 20:
        return np.zeros(n)
    
    # Prior day's high, low, close
    prior_high = daily_ohlc['high'].shift(1).values  # shift(1) for prior day
    prior_low = daily_ohlc['low'].shift(1).values
    prior_close = daily_ohlc['close'].shift(1).values
    
    # Camarilla H3 and L3
    camarilla_h3 = prior_close + 1.1 * (prior_high - prior_low) / 4
    camarilla_l3 = prior_close - 1.1 * (prior_high - prior_low) / 4
    
    # Align to 1h timeframe (wait for daily close)
    camarilla_h3_aligned = align_htf_to_ltf(prices, daily_ohlc, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, daily_ohlc, camarilla_l3)
    
    # 4h EMA34 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Prior 4h EMA34 for trend direction (rising/falling)
    prior_ema_34_4h = np.roll(ema_34_4h_aligned, 1)
    prior_ema_34_4h[0] = ema_34_4h_aligned[0]  # avoid NaN at start
    ema_rising = ema_34_4h_aligned > prior_ema_34_4h
    ema_falling = ema_34_4h_aligned < prior_ema_34_4h
    
    # 1d volume spike confirmation: volume > 1.5x 20-day average
    daily_volume = daily_ohlc['volume'].values
    vol_ma_20 = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = daily_volume > (1.5 * vol_ma_20)
    vol_spike_aligned = align_htf_to_ltf(prices, daily_ohlc, vol_spike)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34) + 1  # need 20d vol MA, 34 EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade during session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above H3 AND uptrend AND volume spike
            if (close[i] > camarilla_h3_aligned[i] and 
                ema_rising[i] and 
                vol_spike_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: break below L3 AND downtrend AND volume spike
            elif (close[i] < camarilla_l3_aligned[i] and 
                  ema_falling[i] and 
                  vol_spike_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: break below L3 OR trend turns down
            if (close[i] < camarilla_l3_aligned[i] or 
                ema_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: break above H3 OR trend turns up
            if (close[i] > camarilla_h3_aligned[i] or 
                ema_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA34_1dVolSpike_Session_v1"
timeframe = "1h"
leverage = 1.0