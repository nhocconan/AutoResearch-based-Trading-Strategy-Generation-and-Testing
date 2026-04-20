#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily price action near weekly VWAP with volume confirmation and ADX trend filter
# Weekly VWAP acts as institutional reference point; price reverts to mean in range, breaks out in trend
# Works in bull/bear: mean reversion in range, trend following in strong moves
# Target: 20-50 trades/year to minimize fee drag

name = "1d_VWAP_MeanReversion_Trend_ADX_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Weekly VWAP calculation ===
    # Typical price * volume cumulative / volume cumulative
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    pv = (typical_price * df_1w['volume']).cumsum()
    vol_cum = df_1w['volume'].cumsum()
    vwap = pv / vol_cum
    vwap_values = vwap.values
    
    # Align VWAP to daily timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap_values)
    
    # === ADX(14) for trend strength on daily ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def smooth_series(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            result[period-1] = np.nansum(values[:period])
            for i in range(period, len(values)):
                result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    atr = smooth_series(tr, 14)
    plus_di = 100 * smooth_series(plus_dm, 14) / atr
    minus_di = 100 * smooth_series(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_series(dx, 14)
    
    # === Volume confirmation ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Get values
        close_val = close[i]
        vwap_val = vwap_aligned[i]
        adx_val = adx[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(vwap_val) or np.isnan(adx_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price below VWAP in low ADX (range) with volume spike
            # Short: Price above VWAP in low ADX (range) with volume spike
            if adx_val < 25:  # Range market
                if close_val < vwap_val and vol_ratio_val > 1.8:
                    signals[i] = 0.25
                    position = 1
                elif close_val > vwap_val and vol_ratio_val > 1.8:
                    signals[i] = -0.25
                    position = -1
            else:  # Trending market
                # Follow trend: buy above VWAP in uptrend, sell below VWAP in downtrend
                if close_val > vwap_val and vol_ratio_val > 1.5:
                    signals[i] = 0.25
                    position = 1
                elif close_val < vwap_val and vol_ratio_val > 1.5:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: Price crosses above VWAP in trend OR ADX weakens
            if close_val > vwap_val and adx_val > 25:
                signals[i] = 0.25  # Stay long
            else:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short exit: Price crosses below VWAP in trend OR ADX weakens
            if close_val < vwap_val and adx_val > 25:
                signals[i] = -0.25  # Stay short
            else:
                signals[i] = 0.0
                position = 0
    
    return signals