#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with 1w HMA filter and volume confirmation
# - Long when 1d price > KAMA(ER=10) AND 1w HMA(21) rising AND 1d volume > 1.5x 20-period average
# - Short when 1d price < KAMA(ER=10) AND 1w HMA(21) falling AND 1d volume > 1.5x 20-period average
# - Exit when price crosses back below/above KAMA OR 1w HMA flattens
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - KAMA adapts to market noise, reducing whipsaw in ranging markets
# - 1w HMA provides higher timeframe trend filter to avoid counter-trend trades
# - Volume confirmation ensures institutional participation

name = "1d_1w_kama_hma_volume_v1"
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
    
    # Pre-compute 1d volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d KAMA ( Kaufman Adaptive Moving Average )
    # KAMA uses Efficiency Ratio (ER) to adapt smoothing constant
    def kama(close, er_period=10, fast_sc=2, slow_sc=30):
        """Kaufman Adaptive Moving Average"""
        # Calculate Change and Volatility
        change = np.abs(np.diff(close, 1))
        change = np.insert(change, 0, 0)  # First element is 0
        
        volatility = np.zeros_like(close)
        for i in range(er_period, len(close)):
            volatility[i] = np.sum(np.abs(np.diff(close[i-er_period+1:i+1])))
        
        # Avoid division by zero
        volatility = np.where(volatility == 0, 1, volatility)
        
        # Efficiency Ratio
        er = np.where(volatility > 0, change / volatility, 0)
        
        # Smoothing Constant
        sc = np.square(er * (fast_sc - slow_sc) + slow_sc)
        
        # KAMA calculation
        kama_values = np.zeros_like(close)
        kama_values[0] = close[0]
        for i in range(1, len(close)):
            kama_values[i] = kama_values[i-1] + sc[i] * (close[i] - kama_values[i-1])
        
        return kama_values
    
    kama_1d = kama(close, er_period=10, fast_sc=2, slow_sc=30)
    
    # Pre-compute 1w HMA (Hull Moving Average)
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    def wma(values, period):
        """Weighted Moving Average"""
        if len(values) < period:
            return np.full_like(values, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    def hma(values, period):
        """Hull Moving Average"""
        if len(values) < period:
            return np.full_like(values, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        wma_full = wma(values, period)
        wma_half = wma(values, half_period)
        
        # 2 * WMA(n/2) - WMA(n)
        raw_hma = 2 * wma_half - wma_full
        
        # WMA(sqrt(n)) of the result
        hma_values = wma(raw_hma, sqrt_period)
        
        # Pad with NaN to match original length
        result = np.full_like(values, np.nan)
        result[period-1:] = hma_values
        return result
    
    close_1w = df_1w['close'].values
    hma_1w = hma(close_1w, 21)
    
    # Calculate HMA slope (rising/falling)
    hma_slope = np.diff(hma_1w, prepend=hma_1w[0])
    hma_rising = hma_slope > 0
    hma_falling = hma_slope < 0
    
    # Align HTF indicators to 1d timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1w, hma_rising)
    hma_falling_aligned = align_htf_to_ltf(prices, df_1w, hma_falling)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(kama_1d[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(hma_rising_aligned[i]) or np.isnan(hma_falling_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price > KAMA AND 1w HMA rising AND volume spike
            if (close[i] > kama_1d[i] and 
                hma_rising_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price < KAMA AND 1w HMA falling AND volume spike
            elif (close[i] < kama_1d[i] and 
                  hma_falling_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses back below/above KAMA OR 1w HMA flattens
            exit_long = (position == 1 and (close[i] <= kama_1d[i] or not hma_rising_aligned[i]))
            exit_short = (position == -1 and (close[i] >= kama_1d[i] or not hma_falling_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals