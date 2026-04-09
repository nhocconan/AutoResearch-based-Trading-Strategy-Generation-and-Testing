#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams Fractals with volume confirmation and ADX regime filter
# Williams Fractals identify key swing points where price reverses
# Long when price breaks above recent bearish fractal with volume confirmation in trending regime (ADX > 25)
# Short when price breaks below recent bullish fractal with volume confirmation in trending regime
# In ranging regime (ADX < 20), fade the fractal extremes: long at bullish fractal, short at bearish fractal
# Uses discrete position sizing 0.25 to target ~20-50 trades/year and minimize fee drag
# Works in bull/bear markets: breakout follows momentum in trending regimes, mean reversion at fractals in ranging regimes

name = "4h_1d_williams_fractal_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Fractals on 1d data
    # Bearish fractal: high[n] > high[n-2] and high[n] > high[n-1] and high[n] > high[n+1] and high[n] > high[n+2]
    # Bullish fractal: low[n] < low[n-2] and low[n] < low[n-1] and low[n] < low[n+1] and low[n] < low[n+2]
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i-1] and 
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        if (low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i-1] and 
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Williams Fractals need 2 extra bars for confirmation (the two bars after the center)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 1d ADX(14) for regime filter
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full(len(high), np.nan)
        
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smoothed TR, DM+
        def wilders_smooth(values, period):
            if len(values) < period:
                return np.full(len(values), np.nan)
            alpha = 1.0 / period
            result = np.full(len(values), np.nan)
            result[period-1] = np.nanmean(values[:period])
            for i in range(period, len(values)):
                result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
            return result
        
        atr = wilders_smooth(tr, period)
        dm_plus_smooth = wilders_smooth(dm_plus, period)
        dm_minus_smooth = wilders_smooth(dm_minus, period)
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        
        # DX and ADX
        dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
        dx = np.where((di_plus + di_minus) != 0, dx, 0)
        
        adx = wilders_smooth(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1d average volume (20-period) for volume confirmation
    if 'volume' in df_1d.columns:
        volume_1d = df_1d['volume'].values
    else:
        volume_1d = np.ones_like(close_1d)  # fallback
    
    vol_s_1d = pd.Series(volume_1d)
    avg_vol_1d = vol_s_1d.rolling(window=20, min_periods=20).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        trending_regime = adx_1d_aligned[i] > 25
        ranging_regime = adx_1d_aligned[i] < 20
        
        if position == 1:  # Long position
            if trending_regime and volume_confirmed:
                # Exit long if price falls below recent bearish fractal
                if close[i] < bearish_fractal_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if price moves back above bullish fractal (mean reversion exit)
                if close[i] > bullish_fractal_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime and volume_confirmed:
                # Exit short if price rises above recent bullish fractal
                if close[i] > bullish_fractal_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if price moves back below bearish fractal (mean reversion exit)
                if close[i] < bearish_fractal_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime and volume_confirmed:
                # Breakout strategy in trending market
                if close[i] > bearish_fractal_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < bullish_fractal_aligned[i]:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean reversion at fractal extremes in ranging market
                if close[i] < bullish_fractal_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > bearish_fractal_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals