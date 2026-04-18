# 4h_Combined_Strategy_V2
# Strategy: 4h timeframe with 4h VWAP, 4h ATR, 1h trend filter, and volume confirmation
# Logic: Long when price > VWAP, price > close[1] + ATR*0.5, volume > 1.5x 20-period average, and 1h EMA20 > EMA50
# Short when price < VWAP, price < close[1] - ATR*0.5, volume > 1.5x 20-period average, and 1h EMA20 < EMA50
# Exit: Opposite signal or trend reversal
# Risk: No explicit stop - relies on signal reversal
# Position sizing: 0.25 (25% of capital)
# Target: ~20-30 trades/year per symbol to minimize fee drag

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h VWAP
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = vwap_num / vwap_den
    
    # Calculate 4h ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h volume moving average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1h data for trend filter
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    
    # Calculate 1h EMA20 and EMA50
    ema_20_1h = pd.Series(close_1h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_1h = pd.Series(close_1h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1h indicators to 4h timeframe
    ema_20_aligned = align_htf_to_ltf(prices, df_1h, ema_20_1h)
    ema_50_aligned = align_htf_to_ltf(prices, df_1h, ema_50_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA50 and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions from 1h
        uptrend = ema_20_aligned[i] > ema_50_aligned[i]
        downtrend = ema_20_aligned[i] < ema_50_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Price momentum conditions
        price_above_vwap = close[i] > vwap[i]
        price_below_vwap = close[i] < vwap[i]
        price_momentum_up = close[i] > close[i-1] + atr[i] * 0.5
        price_momentum_down = close[i] < close[i-1] - atr[i] * 0.5
        
        if position == 0:
            # Long: uptrend + volume + price above VWAP + upward momentum
            if uptrend and vol_confirm and price_above_vwap and price_momentum_up:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + price below VWAP + downward momentum
            elif downtrend and vol_confirm and price_below_vwap and price_momentum_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change, volume confirmation, or price below VWAP with downward momentum
            if not uptrend or (vol_confirm and price_below_vwap and price_momentum_down):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change, volume confirmation, or price above VWAP with upward momentum
            if not downtrend or (vol_confirm and price_above_vwap and price_momentum_up):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Combined_Strategy_V2"
timeframe = "4h"
leverage = 1.0
#%%