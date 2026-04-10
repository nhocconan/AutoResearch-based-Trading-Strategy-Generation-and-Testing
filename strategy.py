#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + VWAP mean reversion with 1w trend filter
# - Elder Ray: Bull Power = Close - EMA13, Bear Power = EMA13 - Close
# - Mean reversion: Long when Bull Power < -0.5*ATR AND price < VWAP AND weekly uptrend
# - Short when Bear Power < -0.5*ATR AND price > VWAP AND weekly downtrend
# - Exit when price crosses VWAP or Elder Power reverses
# - Weekly trend filter ensures alignment with major trend
# - VWAP provides dynamic mean reversion target
# - Targets 12-25 trades/year (50-100 total over 4 years) to avoid fee drag
# - Combines mean reversion with trend filter to work in both bull and bear markets

name = "6h_1w_elder_ray_vwap_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Pre-compute ATR(14) for volatility
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute EMA(13) for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute VWAP (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    vp = typical_price * prices['volume'].values
    cum_vp = np.cumsum(vp)
    cum_vol = np.cumsum(prices['volume'].values)
    vwap = np.where(cum_vol > 0, cum_vp / cum_vol, typical_price)
    
    # Pre-compute Elder Ray components
    bull_power = close - ema13  # Buy pressure
    bear_power = ema13 - close  # Sell pressure
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if any required data is invalid
        if (np.isnan(atr[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vwap[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for mean reversion entries
            # Long setup: oversold (strong bear power) AND price below VWAP AND weekly uptrend
            if (bear_power[i] > 0.5 * atr[i] and 
                close[i] < vwap[i] and 
                close[i] > ema50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short setup: overbought (strong bull power) AND price above VWAP AND weekly downtrend
            elif (bull_power[i] > 0.5 * atr[i] and 
                  close[i] > vwap[i] and 
                  close[i] < ema50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit when price crosses VWAP (mean reversion complete)
            # OR when Elder Power reverses (momentum shifts against position)
            if position == 1:  # Long position
                if (close[i] > vwap[i] or  # Price crossed above VWAP
                    bull_power[i] > 0.3 * atr[i]):  # Bull power weakening
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (close[i] < vwap[i] or  # Price crossed below VWAP
                    bear_power[i] > 0.3 * atr[i]):  # Bear power weakening
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals