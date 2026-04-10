#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level AND 1w EMA(21) is rising AND volume > 1.5x 20-period average volume
# - Short when price breaks below Camarilla L3 level AND 1w EMA(21) is falling AND volume > 1.5x 20-period average volume
# - Exit when price crosses back inside the Camarilla H3-L3 range
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Camarilla pivots identify key intraday support/resistance levels
# - Weekly EMA filter ensures we trade with the higher timeframe trend
# - Volume confirmation reduces false breakouts

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Pre-compute 1d OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 1d Camarilla pivot levels (based on previous day)
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low)
    #            L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # First bar uses current values
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Pre-compute 1d volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1w EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_rising = ema_21_1w > np.roll(ema_21_1w, 1)  # Current EMA > previous EMA
    ema_rising[0] = False  # First bar has no previous
    
    # Align HTF indicators to 1d timeframe
    ema_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_rising)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_rising_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND weekly EMA rising AND volume spike
            if (close[i] > camarilla_h3[i] and 
                ema_rising_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below L3 AND weekly EMA falling AND volume spike
            elif (close[i] < camarilla_l3[i] and 
                  not ema_rising_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses back inside the H3-L3 range
            exit_long = (position == 1 and close[i] < camarilla_h3[i])
            exit_short = (position == -1 and close[i] > camarilla_l3[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals