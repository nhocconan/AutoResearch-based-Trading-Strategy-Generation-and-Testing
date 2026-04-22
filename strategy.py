#!/usr/bin/env python3
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
    
    # Load weekly data for trend filter and monthly data for volatility regime
    df_1w = get_htf_data(prices, '1w')
    df_1m = get_htf_data(prices, '1M')
    
    close_1w = df_1w['close'].values
    close_1m = df_1m['close'].values
    
    # Weekly EMA200 for trend filter (bull/bear regime)
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Monthly ATR for volatility regime (high/low volatility filter)
    high_1m = df_1m['high'].values
    low_1m = df_1m['low'].values
    tr1m = np.maximum(high_1m - low_1m, 
                      np.absolute(np.subtract(high_1m, np.roll(close_1m, 1))),
                      np.absolute(np.subtract(low_1m, np.roll(close_1m, 1))))
    atr1m = pd.Series(tr1m).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr1m_pct = atr1m / close_1m  # ATR as percentage of price
    
    # Daily Donchian channel (20-period) for breakout signals
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels (using rolling window)
    upper20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > 1.5 * vol_ma20  # 1.5x volume surge for confirmation
    
    # Align all indicators to daily timeframe
    upper20_aligned = align_htf_to_ltf(prices, df_1d, upper20)
    lower20_aligned = align_htf_to_ltf(prices, df_1d, lower20)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1d, ema200_1w)
    atr1m_pct_aligned = align_htf_to_ltf(prices, df_1d, atr1m_pct)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):  # Start after warmup for indicators
        # Skip if data not ready
        if (np.isnan(upper20_aligned[i]) or np.isnan(lower20_aligned[i]) or
            np.isnan(ema200_1w_aligned[i]) or np.isnan(atr1m_pct_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filters:
        # 1. Trend filter: only take longs in bull market (price > weekly EMA200)
        #    only take shorts in bear market (price < weekly EMA200)
        bull_market = close[i] > ema200_1w_aligned[i]
        bear_market = close[i] < ema200_1w_aligned[i]
        
        # 2. Volatility filter: avoid extremely low volatility (choppy) and extremely high volatility (panic)
        #    Optimal range: 0.5% to 3.0% daily ATR
        vol_ok = (atr1m_pct_aligned[i] >= 0.005) & (atr1m_pct_aligned[i] <= 0.030)
        
        if position == 0:
            # Long: Price breaks above Donchian upper + bull market + vol OK + volume surge
            if (close[i] > upper20_aligned[i] and 
                bull_market and 
                vol_ok and 
                vol_surge[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower + bear market + vol OK + volume surge
            elif (close[i] < lower20_aligned[i] and 
                  bear_market and 
                  vol_ok and 
                  vol_surge[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions:
            # 1. Price returns to opposite Donchian level
            # 2. Trend changes (price crosses weekly EMA200)
            # 3. Volatility becomes too low (< 0.3%) indicating chop
            if position == 1:
                if (close[i] < lower20_aligned[i] or 
                    close[i] < ema200_1w_aligned[i] or
                    atr1m_pct_aligned[i] < 0.003):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > upper20_aligned[i] or 
                    close[i] > ema200_1w_aligned[i] or
                    atr1m_pct_aligned[i] < 0.003):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_WeeklyEMA200Trend_MonthlyATR_Volume"
timeframe = "1d"
leverage = 1.0