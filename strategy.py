#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot long strategy with 1w EMA trend filter and volume confirmation
# - Long when price touches Camarilla L3 support level AND 1w EMA(21) is rising AND volume > 1.5x 20-period average volume
# - Exit when price reaches Camarilla H3 resistance level or closes below L3
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Camarilla pivots identify intraday support/resistance with high probability reversal zones
# - 1w EMA filter ensures we only take longs in the primary weekly uptrend
# - Volume confirmation reduces false signals

name = "1d_1w_camarilla_pivot_trend_volume_v1"
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
    open_price = prices['open'].values
    
    # Pre-compute 20-period average volume for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1w EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_rising = np.gradient(ema_21_1w) > 0  # Rising when slope positive
    
    # Align HTF indicators to 1d timeframe
    ema_21_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_21_rising)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(vol_ma[i]) or np.isnan(ema_21_rising_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Calculate Camarilla pivot levels for today
            # Based on previous day's OHLC
            if i > 0:
                pc = close[i-1]  # Previous close
                ph = high[i-1]   # Previous high
                pl = low[i-1]    # Previous low
                
                # Camarilla levels
                range_val = ph - pl
                if range_val > 0:  # Avoid division by zero
                    camarilla_h3 = pc + (range_val * 1.1 / 4)
                    camarilla_l3 = pc - (range_val * 1.1 / 4)
                    camarilla_h4 = pc + (range_val * 1.1 / 2)
                    camarilla_l4 = pc - (range_val * 1.1 / 2)
                    
                    # Long conditions: price touches L3 support AND weekly EMA rising AND volume spike
                    # Touch defined as low <= L3 and close > L3 (bounce off support)
                    if (low[i] <= camarilla_l3 and 
                        close[i] > camarilla_l3 and 
                        ema_21_rising_aligned[i] and 
                        volume_spike[i]):
                        position = 1
                        signals[i] = 0.25
                    # Short conditions: price touches H3 resistance AND weekly EMA rising AND volume spike
                    # Touch defined as high >= H3 and close < H3 (rejection at resistance)
                    elif (high[i] >= camarilla_h3 and 
                          close[i] < camarilla_h3 and 
                          ema_21_rising_aligned[i] and 
                          volume_spike[i]):
                        position = -1
                        signals[i] = -0.25
                    else:
                        signals[i] = 0.0
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            if i > 0:
                pc = close[i-1]
                ph = high[i-1]
                pl = low[i-1]
                range_val = ph - pl
                
                if range_val > 0:
                    camarilla_h3 = pc + (range_val * 1.1 / 4)
                    camarilla_l3 = pc - (range_val * 1.1 / 4)
                    
                    if position == 1:  # Long position
                        # Exit when price reaches H3 resistance or closes below L3 support
                        exit_long = (high[i] >= camarilla_h3) or (close[i] < camarilla_l3)
                        if exit_long:
                            position = 0
                            signals[i] = 0.0
                        else:
                            signals[i] = 0.25
                    else:  # Short position
                        # Exit when price reaches L3 support or closes above H3 resistance
                        exit_short = (low[i] <= camarilla_l3) or (close[i] > camarilla_h3)
                        if exit_short:
                            position = 0
                            signals[i] = 0.0
                        else:
                            signals[i] = -0.25
                else:
                    if position == 1:
                        signals[i] = 0.25
                    else:
                        signals[i] = -0.25
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals