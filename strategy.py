#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R1/S1 breakout with volume confirmation and 1d ADX trend filter.
# Long when price breaks above R1 AND volume > 1.5x daily average volume AND ADX > 25 (trending market)
# Short when price breaks below S1 AND volume > 1.5x daily average volume AND ADX > 25
# Exit when price crosses back through the pivot point (PP)
# Uses Camarilla for precise intraday levels, volume for confirmation, ADX to avoid ranging markets.
# Target: 20-30 trades/year per symbol.
name = "4h_Camarilla_R1S1_Volume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels (based on previous day)
    # PP = (H + L + C) / 3
    # R1 = PP + (H - L) * 1.1 / 12
    # S1 = PP - (H - L) * 1.1 / 12
    pp = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    r1 = pp + (df_1d['high'] - df_1d['low']) * 1.1 / 12
    s1 = pp - (df_1d['high'] - df_1d['low']) * 1.1 / 12
    
    # Calculate ADX (14-period)
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_roll = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    plus_dm_roll = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum()
    minus_dm_roll = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_roll / tr_roll
    minus_di = 100 * minus_dm_roll / tr_roll
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align all 1d indicators to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Get 1d average volume for confirmation
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pp_val = pp_aligned[i]
        
        # Trend filter: only trade when ADX > 25 (trending market)
        trending = adx_val > 25
        
        if position == 0:
            # Long entry: break above R1 + volume spike + trending market
            if price > r1_val and vol > 1.5 * vol_ma and trending:
                signals[i] = 0.25
                position = 1
            # Short entry: break below S1 + volume spike + trending market
            elif price < s1_val and vol > 1.5 * vol_ma and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below pivot point
            if price < pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above pivot point
            if price > pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R1/S1 breakout with volume confirmation and 1d ADX trend filter.
# Long when price breaks above R1 AND volume > 1.5x daily average volume AND ADX > 25 (trending market)
# Short when price breaks below S1 AND volume > 1.5x daily average volume AND ADX > 25
# Exit when price crosses back through the pivot point (PP)
# Uses Camarilla for precise intraday levels, volume for confirmation, ADX to avoid ranging markets.
# Target: 20-30 trades/year per symbol.
name = "4h_Camarilla_R1S1_Volume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels (based on previous day)
    # PP = (H + L + C) / 3
    # R1 = PP + (H - L) * 1.1 / 12
    # S1 = PP - (H - L) * 1.1 / 12
    pp = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    r1 = pp + (df_1d['high'] - df_1d['low']) * 1.1 / 12
    s1 = pp - (df_1d['high'] - df_1d['low']) * 1.1 / 12
    
    # Calculate ADX (14-period)
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_roll = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    plus_dm_roll = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum()
    minus_dm_roll = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_roll / tr_roll
    minus_di = 100 * minus_dm_roll / tr_roll
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align all 1d indicators to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Get 1d average volume for confirmation
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pp_val = pp_aligned[i]
        
        # Trend filter: only trade when ADX > 25 (trending market)
        trending = adx_val > 25
        
        if position == 0:
            # Long entry: break above R1 + volume spike + trending market
            if price > r1_val and vol > 1.5 * vol_ma and trending:
                signals[i] = 0.25
                position = 1
            # Short entry: break below S1 + volume spike + trending market
            elif price < s1_val and vol > 1.5 * vol_ma and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below pivot point
            if price < pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above pivot point
            if price > pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals