#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume spike confirmation
# - Long when price is above Ichimoku cloud (Senkou Span A & B) AND 1d close > 1d EMA50 (bullish regime) AND volume > 1.5x 20-period average
# - Short when price is below Ichimoku cloud AND 1d close < 1d EMA50 (bearish regime) AND volume > 1.5x 20-period average
# - Exit when price crosses Tenkan-Kijun (TK) line in opposite direction
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Ichimoku cloud provides dynamic support/resistance that adapts to volatility
# - 1d EMA50 filter ensures we trade with higher timeframe trend
# - Volume confirmation reduces false breakouts

name = "6h_1d_ichimoku_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 6h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 6h Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Pre-compute 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    bullish_regime_1d = close_1d > ema50_1d
    bearish_regime_1d = close_1d < ema50_1d
    
    # Align HTF indicators to 6h timeframe
    bullish_regime_aligned = align_htf_to_ltf(prices, df_1d, bullish_regime_1d)
    bearish_regime_aligned = align_htf_to_ltf(prices, df_1d, bearish_regime_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(bullish_regime_aligned[i]) or np.isnan(bearish_regime_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Determine cloud boundaries (top and bottom of cloud)
        cloud_top = np.maximum(senkou_a[i], senkou_b[i])
        cloud_bottom = np.minimum(senkou_a[i], senkou_b[i])
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price above cloud AND bullish 1d regime AND volume spike
            if (close[i] > cloud_top and 
                bullish_regime_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price below cloud AND bearish 1d regime AND volume spike
            elif (close[i] < cloud_bottom and 
                  bearish_regime_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit on TK cross
            # Exit when price crosses Tenkan-Kijun line in opposite direction
            exit_long = (position == 1 and close[i] < tenkan[i])
            exit_short = (position == -1 and close[i] > kijun[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals