#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d/1w HTF context using volume-weighted RSI (VW-RSI)
# to identify momentum extremes, filtered by volatility regime (ATR-based) and
# volume confirmation. VW-RSI combines price momentum with institutional activity.
# Works in bull/bear by adapting to volatility regimes and using volume as
# confirmation of institutional participation, reducing false signals.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for HTF context
    daily = get_htf_data(prices, '1d')
    
    # Calculate ATR for volatility regime filtering
    tr1 = daily['high'].values[1:] - daily['low'].values[1:]
    tr2 = np.abs(daily['high'].values[1:] - daily['close'].values[:-1])
    tr3 = np.abs(daily['low'].values[1:] - daily['close'].values[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # Volatility regime: high volatility (trending) when ATR > 20-period MA of ATR
    atr_ma_20 = pd.Series(atr_14d).rolling(window=20, min_periods=20).mean().values
    atr_ma_20_aligned = align_htf_to_ltf(prices, daily, atr_ma_20)
    high_vol_regime = atr_14d_aligned > atr_ma_20_aligned
    
    # Calculate volume-weighted RSI (VW-RSI) on daily data
    # VW-RSI = 100 - (100 / (1 + RS)), where RS = avg gains / avg losses weighted by volume
    price_change = np.diff(daily['close'].values, prepend=daily['close'].values[0])
    gains = np.where(price_change > 0, price_change, 0)
    losses = np.where(price_change < 0, -price_change, 0)
    
    # Volume-weighted gains and losses
    vol_weights = daily['volume'].values
    vol_gains = gains * vol_weights
    vol_losses = losses * vol_weights
    
    # Smoothed volume-weighted RS
    avg_vol_gain = pd.Series(vol_gains).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_vol_loss = pd.Series(vol_losses).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_vol_gain, avg_vol_loss, out=np.full_like(avg_vol_gain, 50.0), where=avg_vol_loss!=0)
    vw_rsi = 100 - (100 / (1 + rs))
    vw_rsi_aligned = align_htf_to_ltf(prices, daily, vw_rsi)
    
    # Volume confirmation: current volume > 1.3x 20-day average volume
    vol_ma_20 = pd.Series(daily['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, daily, vol_ma_20)
    vol_confirmation = volume > (1.3 * vol_ma_20_aligned)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(vw_rsi_aligned[i]) or np.isnan(atr_14d_aligned[i]) or 
            np.isnan(atr_ma_20_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            continue
        
        # Only trade in high volatility regimes (trending markets)
        if not high_vol_regime[i]:
            signals[i] = 0.0
            continue
            
        # Long: VW-RSI oversold (<30) in uptrend + volume confirmation
        if (vw_rsi_aligned[i] < 30 and 
            close[i] > close[i-1] and  # minor uptrend confirmation
            vol_confirmation[i]):
            signals[i] = 0.25
        
        # Short: VW-RSI overbought (>70) in downtrend + volume confirmation
        elif (vw_rsi_aligned[i] > 70 and 
              close[i] < close[i-1] and  # minor downtrend confirmation
              vol_confirmation[i]):
            signals[i] = -0.25
        
        # Exit: VW-RSI returns to neutral zone (40-60)
        elif 40 <= vw_rsi_aligned[i] <= 60:
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_VWRSI_Volatility_Regime_Volume"
timeframe = "12h"
leverage = 1.0