# [73930] Hypothesis: 1d timeframe with 1h HTF trend filter + volume spike + volatility regime
# Uses 1h EMA50 for trend, 1d close > EMA50 for long bias, < EMA50 for short bias
# Entry on 1d close breaking ATR-based bands with volume confirmation
# Volatility regime filter: only trade when ATR(14) > ATR(50) (expanding volatility)
# Target: 50-100 trades over 4 years (~12-25/year) to avoid fee drag
# Works in bull/bear: trend filter adapts, volatility regime avoids chop

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # 1h data for trend filter (HTF relative to 1d)
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    
    # 1h EMA50 for trend
    ema_50_1h = pd.Series(close_1h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_50_1h)
    
    # 1d ATR for volatility bands and regime
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Volatility regime: expanding volatility (trending market)
    vol_expanding = atr_14 > atr_50
    
    # ATR-based bands (1.5 * ATR)
    upper_band = close + 1.5 * atr_14
    lower_band = close - 1.5 * atr_14
    
    # Volume filter
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 1.5 * vol_ma20  # Moderate volume surge
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1h_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(atr_50[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > upper band + volume surge + expanding vol + above 1h EMA50
            if (close[i] > upper_band[i] and vol_surge[i] and vol_expanding[i] and 
                close[i] > ema_50_1h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < lower band + volume surge + expanding vol + below 1h EMA50
            elif (close[i] < lower_band[i] and vol_surge[i] and vol_expanding[i] and 
                  close[i] < ema_50_1h_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Close crosses back through mid-price or volatility contracts
            mid_price = (upper_band[i] + lower_band[i]) / 2
            if position == 1:
                if close[i] < mid_price[i] or not vol_expanding[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > mid_price[i] or not vol_expanding[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_VolatilityBreakout_VolumeSurge_EMA50Trend_v1"
timeframe = "1d"
leverage = 1.0