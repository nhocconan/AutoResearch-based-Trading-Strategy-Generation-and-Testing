#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot long/short with 1w trend filter and volume confirmation
# - Long when price > Camarilla H3 (1d) AND 1w close > 1w open (bullish weekly candle) AND volume > 1.5x 20-day average volume
# - Short when price < Camarilla L3 (1d) AND 1w close < 1w open (bearish weekly candle) AND volume > 1.5x 20-day average volume
# - Exit when price crosses Camarilla H4/L4 levels OR weekly trend reverses
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Camarilla pivots identify intraday support/resistance levels that work in ranging and trending markets
# - Weekly trend filter ensures we trade with the higher timeframe momentum
# - Volume confirmation reduces false breakouts

name = "1d_1w_camarilla_pivot_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Pre-compute 1d OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Pre-compute 1d volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d Camarilla pivot levels (based on previous day's range)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.25 * (high - low)
    # H2 = close + 1.1 * (high - low)
    # H1 = close + 0.825 * (high - low)
    # L1 = close - 0.825 * (high - low)
    # L2 = close - 1.1 * (high - low)
    # L3 = close - 1.25 * (high - low)
    # L4 = close - 1.5 * (high - low)
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    
    # Set first day's values to zero (no previous day)
    high_prev[0] = high[0]
    low_prev[0] = low[0]
    close_prev[0] = close[0]
    
    daily_range = high_prev - low_prev
    
    camarilla_h3 = close_prev + 1.25 * daily_range
    camarilla_l3 = close_prev - 1.25 * daily_range
    camarilla_h4 = close_prev + 1.5 * daily_range
    camarilla_l4 = close_prev - 1.5 * daily_range
    
    # Pre-compute 1w trend (bullish/bearish weekly candle)
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open  # Bullish weekly candle
    weekly_bearish = weekly_close < weekly_open  # Bearish weekly candle
    
    # Align HTF indicators to 1d timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup for volume MA
        # Skip if any required data is invalid
        if (np.isnan(vol_ma[i]) or np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price > Camarilla H3 AND bullish weekly candle AND volume spike
            if (close[i] > camarilla_h3[i] and 
                weekly_bullish_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price < Camarilla L3 AND bearish weekly candle AND volume spike
            elif (close[i] < camarilla_l3[i] and 
                  weekly_bearish_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses Camarilla H4/L4 OR weekly trend reverses
            exit_long = (position == 1 and 
                        (close[i] >= camarilla_h4[i] or not weekly_bullish_aligned[i]))
            exit_short = (position == -1 and 
                         (close[i] <= camarilla_l4[i] or not weekly_bearish_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals