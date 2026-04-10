#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Williams Alligator with 1d trend filter and volume confirmation
# - ADX(14) > 25 identifies trending markets
# - Williams Alligator: Jaw(13), Teeth(8), Lips(5) SMAs of median price
# - Long when Alligator is bullish (Lips > Teeth > Jaw) AND ADX rising AND 1d close > EMA50 AND volume > 1.5x avg
# - Short when Alligator is bearish (Lips < Teeth < Jaw) AND ADX rising AND 1d close < EMA50 AND volume > 1.5x avg
# - Exit when Alligator alignment breaks or ADX falls below 20
# - Uses discrete position sizing (0.25) to control drawdown
# - Targets ~15-25 trades/year (60-100 total over 4 years) to avoid fee drag
# - Williams Alligator catches trends early; ADX filters for strength; volume confirms conviction
# - Works in both bull (strong uptrend) and bear (strong downtrend) markets via Alligator alignment
# - 1d EMA50 filter ensures alignment with higher timeframe trend

name = "6h_1d_adx_alligator_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Williams Alligator components (using 6h data)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    median_price = (high + low) / 2.0
    
    # Alligator: Jaw(13), Teeth(8), Lips(5) - all SMAs of median price
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Pre-compute ADX(14)
    # ADX requires +DI, -DI, and TR
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(adx[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: Alligator bullish (Lips > Teeth > Jaw) AND ADX > 25 AND ADX rising AND 1d uptrend AND volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and  # Bullish alignment
                adx[i] > 25 and 
                adx[i] > adx[i-1] and  # ADX rising
                close[i] > ema50_1d_aligned[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: Alligator bearish (Lips < Teeth < Jaw) AND ADX > 25 AND ADX rising AND 1d downtrend AND volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and  # Bearish alignment
                  adx[i] > 25 and 
                  adx[i] > adx[i-1] and  # ADX rising
                  close[i] < ema50_1d_aligned[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Alligator alignment breaks (long exits when Lips <= Teeth or Teeth <= Jaw)
            # 2. ADX falls below 20 (trend weakening)
            if position == 1:
                if (lips[i] <= teeth[i] or teeth[i] <= jaw[i]) or adx[i] < 20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:
                if (lips[i] >= teeth[i] or teeth[i] >= jaw[i]) or adx[i] < 20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals