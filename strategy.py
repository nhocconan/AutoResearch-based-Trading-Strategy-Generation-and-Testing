#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d ATR-based volatility filter and volume spike
# In high volatility regimes (expanding ATR), trade breakouts of 4h Donchian(20) with volume confirmation
# In low volatility regimes (contracting ATR), fade moves using 4h RSI(14) extremes with 4h SMA(50) filter
# This adapts to both trending and ranging markets while minimizing false breakouts
# Uses discrete position sizing 0.25 to target ~30-60 trades/year and minimize fee drag

name = "4h_1d_atr_vol_regime_v1"
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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones_like(close_1d)
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    atr_ma_10_1d = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
    
    # Volatility regime: high vol when current ATR > 1.2x 10-period MA (expanding volatility)
    vol_regime_high = atr_1d > 1.2 * atr_ma_10_1d
    vol_regime_low = atr_1d < 0.8 * atr_ma_10_1d  # contracting volatility
    
    # Calculate 4h Donchian channels (20-period)
    def donchian_channels(high_arr, low_arr, period):
        upper = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_hi, donch_lo = donchian_channels(high, low, 20)
    
    # Calculate 4h RSI(14) for mean reversion signals
    def rsi_wilder(close_arr, period):
        delta = np.diff(close_arr, prepend=close_arr[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        def Wilder(values, period):
            if len(values) < period:
                return np.full(len(values), np.nan)
            alpha = 1.0 / period
            result = np.full(len(values), np.nan)
            result[period-1] = np.nanmean(values[:period])
            for i in range(period, len(values)):
                result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
            return result
        
        avg_gain = Wilder(gain, period)
        avg_loss = Wilder(loss, period)
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_4h = rsi_wilder(close, 14)
    
    # Calculate 4h SMA(50) for trend filter
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 4h average volume (20-period) for volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    vol_regime_high_aligned = align_htf_to_ltf(prices, df_1d, vol_regime_high)
    vol_regime_low_aligned = align_htf_to_ltf(prices, df_1d, vol_regime_low)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_hi[i]) or np.isnan(donch_lo[i]) or 
            np.isnan(rsi_4h[i]) or np.isnan(sma_50[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(vol_regime_high_aligned[i]) or
            np.isnan(vol_regime_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x average volume
        volume_confirmed = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 1:  # Long position
            if vol_regime_high_aligned[i] and volume_confirmed:
                # Exit long if price falls below Donchian lower
                if close[i] < donch_lo[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif vol_regime_low_aligned[i]:
                # Exit long if RSI moves back above 50 (mean reversion exit)
                if rsi_4h[i] > 50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if vol_regime_high_aligned[i] and volume_confirmed:
                # Exit short if price rises above Donchian upper
                if close[i] > donch_hi[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif vol_regime_low_aligned[i]:
                # Exit short if RSI moves back below 50 (mean reversion exit)
                if rsi_4h[i] < 50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if vol_regime_high_aligned[i] and volume_confirmed:
                # Breakout strategy in high volatility (trending) regime
                if close[i] > donch_hi[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < donch_lo[i]:
                    position = -1
                    signals[i] = -0.25
            elif vol_regime_low_aligned[i]:
                # Mean reversion strategy in low volatility (ranging) regime
                if rsi_4h[i] < 30 and close[i] > sma_50[i]:  # Oversold but above SMA50 (avoid catching falling knife)
                    position = 1
                    signals[i] = 0.25
                elif rsi_4h[i] > 70 and close[i] < sma_50[i]:  # Overbought but below SMA50 (avoid catching rising knife)
                    position = -1
                    signals[i] = -0.25
    
    return signals