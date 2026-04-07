#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h trend filter and volume confirmation
# Williams %R measures overbought/oversold levels: long when %R crosses above -80 from below in 12h uptrend
# short when %R crosses below -20 from above in 12h downtrend
# Volume confirmation: current volume > 1.5x 20-period average
# Exit when Williams %R reverses across -50 level
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Target: 80-120 total trades over 4 years (20-30/year)

name = "6h_williamsr_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R(14) calculation
    def calculate_williams_r(high, low, close, period=14):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr.fillna(0).values
    
    wr = calculate_williams_r(high, low, close, 14)
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA50 for trend
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume average for confirmation
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(wr[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R crosses below -50 (momentum fading)
            elif wr[i] < -50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R crosses above -50 (momentum fading)
            elif wr[i] > -50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Williams %R extremes, trend alignment, and volume confirmation
            # Bullish: Williams %R crosses above -80 from below (oversold bounce) in uptrend
            bullish_setup = wr[i] > -80 and wr[i-1] <= -80
            # Bearish: Williams %R crosses below -20 from above (overbought rejection) in downtrend
            bearish_setup = wr[i] < -20 and wr[i-1] >= -20
            
            # Long: bullish setup, 12h uptrend, volume spike
            if (bullish_setup and
                close[i] > ema50_12h_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bearish setup, 12h downtrend, volume spike
            elif (bearish_setup and
                  close[i] < ema50_12h_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals