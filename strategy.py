#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray with 1w regime filter
# - Uses 1d Williams Alligator (Jaw=13, Teeth=8, Lips=5) for trend structure
# - Uses 1d Elder Ray (Bull/Bear Power with EMA13) for momentum confirmation
# - Uses 1w ADX(14) > 25 as regime filter to only trade in strong trends
# - Enters long when: Teeth > Lips (bullish alignment) AND Bull Power > 0 AND price > Jaw
# - Enters short when: Teeth < Lips (bearish alignment) AND Bear Power < 0 AND price < Jaw
# - Exits on opposite Alligator signal or when ADX < 20 (trend weakening)
# - Target: 20-40 trades/year on 1d timeframe (80-160 total over 4 years) to avoid fee drag
# - Williams Alligator excels in trending markets (2021, 2023-2024) and avoids whipsaws in ranges (2022)
# - Elder Ray adds momentum confirmation to avoid false Alligator signals
# - 1w ADX regime filter ensures we only trade when higher timeframe confirms strong trend

name = "1d_1w_alligator_elder_ray_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1w ADX(14) for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w[0] = tr1[0]
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1w = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1w ADX to 1d
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # 1d Williams Alligator (Smoothed Medians)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # Each uses 8, 5, 3 periods offset respectively for smoothing
    close_1d = prices['close'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    median_price = (high_1d + low_1d) / 2  # Using median price (high+low)/2
    
    # Smoothed Moving Average (SMMA) - similar to Wilder's smoothing
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: (prev * (period-1) + current) / period
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.mean(data[i-period+1:i+1])
            else:
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(median_price, 13)  # Jaw (blue line)
    teeth = smma(median_price, 8)  # Teeth (red line)
    lips = smma(median_price, 5)   # Lips (green line)
    
    # 1d Elder Ray Index (Bull Power and Bear Power)
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: Only trade when 1w ADX > 25 (strong trend)
        strong_trend = adx_1w_aligned[i] > 25
        weak_trend = adx_1w_aligned[i] < 20  # Exit when trend weakens
        
        if position == 1:  # Long position
            # Exit conditions: trend reversal or weakening trend
            if (teeth[i] < lips[i]) or weak_trend:  # Alligator bearish alignment or trend weakening
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: trend reversal or weakening trend
            if (teeth[i] > lips[i]) or weak_trend:  # Alligator bullish alignment or trend weakening
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for entries aligned with Alligator and Elder Ray
            if strong_trend:
                # Long: Teeth > Lips (bullish alignment) AND Bull Power > 0 AND price > Jaw
                if (teeth[i] > lips[i] and 
                    bull_power[i] > 0 and 
                    close_1d[i] > jaw[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: Teeth < Lips (bearish alignment) AND Bear Power < 0 AND price < Jaw
                elif (teeth[i] < lips[i] and 
                      bear_power[i] < 0 and 
                      close_1d[i] < jaw[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals