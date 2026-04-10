#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 12h volume-weighted average price (VWAP) deviation + 1d ATR regime filter
# - Long when price breaks above Donchian(20) high AND price > 12h VWAP AND 1d ATR(14) < 1d ATR(50) (low volatility regime)
# - Short when price breaks below Donchian(20) low AND price < 12h VWAP AND 1d ATR(14) < 1d ATR(50)
# - Exit when price crosses Donchian(20) midline
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Donchian breakouts capture strong momentum moves
# - 12h VWAP filter ensures we trade with the higher timeframe value area
# - 1d ATR regime filter avoids high volatility choppy markets where breakouts fail

name = "6h_12h_1d_donchian_vwap_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 20 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 6h Donchian channels (20)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian high/low (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Pre-compute 12h VWAP
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Typical price
    typical_price_12h = (high_12h + low_12h + close_12h) / 3
    # VWAP = cumulative(typical_price * volume) / cumulative(volume)
    cum_vol_12h = np.cumsum(volume_12h)
    cum_tpv_12h = np.cumsum(typical_price_12h * volume_12h)
    vwap_12h = np.divide(cum_tpv_12h, cum_vol_12h, out=np.full_like(cum_tpv_12h, np.nan), where=cum_vol_12h!=0)
    
    # Align 12h VWAP to 6h timeframe
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    # Pre-compute 1d ATR for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First TR is undefined
    
    # ATR(14) and ATR(50)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # ATR regime: low volatility when ATR(14) < ATR(50)
    atr_regime_low = atr_14 < atr_50
    
    # Align 1d ATR regime to 6h timeframe
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime_low)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vwap_12h_aligned[i]) or np.isnan(atr_regime_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Donchian high AND price > 12h VWAP AND low volatility regime
            if (close[i] > donch_high[i] and 
                close[i] > vwap_12h_aligned[i] and 
                atr_regime_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Donchian low AND price < 12h VWAP AND low volatility regime
            elif (close[i] < donch_low[i] and 
                  close[i] < vwap_12h_aligned[i] and 
                  atr_regime_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses Donchian midline
            exit_long = (position == 1 and close[i] < donch_mid[i])
            exit_short = (position == -1 and close[i] > donch_mid[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals