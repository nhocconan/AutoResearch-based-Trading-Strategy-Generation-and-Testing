#!/usr/bin/env python3
"""
Experiment #5051: 6h Elder Ray + 1d ADX Regime + Volume Confirmation
HYPOTHESIS: On 6h timeframe, Elder Ray (Bull Power/Bear Power) identifies momentum strength while 1d ADX regime filter (ADX>25) ensures trending markets. Volume > 1.5x average confirms participation. This combination works in both bull (Bull Power > 0 in uptrend) and bear (Bear Power < 0 in downtrend) markets by capturing institutional moves with proper regime alignment. Target: 12-37 trades/year on 6h to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5051_6h_elder_ray_1d_adx_regime_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for ADX regime
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: ADX(14) for regime filter ===
    if len(df_1d) >= 14:
        # Calculate True Range
        tr1 = df_1d['high'].values[1:] - df_1d['low'].values[1:]
        tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
        tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
        tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Calculate Directional Movement
        up_move = df_1d['high'].values[1:] - df_1d['high'].values[:-1]
        down_move = df_1d['low'].values[:-1] - df_1d['low'].values[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smooth TR and DM
        tr_14 = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
        plus_dm_14 = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
        minus_dm_14 = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Calculate DI and DX
        plus_di = 100 * plus_dm_14 / tr_14
        minus_di = 100 * minus_dm_14 / tr_14
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX
        adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Align to 6h timeframe
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    else:
        adx_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Elder Ray (Bull Power/Bear Power) ===
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(13, 20)  # EMA13, Volume MA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position when Elder Power reverses or ADX weakens ---
        if in_position:
            exit_long = (position_side > 0) and ((bull_power[i] <= 0) or (adx_aligned[i] < 20))
            exit_short = (position_side < 0) and ((bear_power[i] >= 0) or (adx_aligned[i] < 20))
            
            if exit_long or exit_short:
                in_position = False
                position_side = 0
                signals[i] = 0.0
            else:
                signals[i] = SIZE * position_side
            continue
        
        # --- New Position Entry Logic ---
        # Regime filter: ADX > 25 indicates trending market
        trending_regime = adx_aligned[i] > 25
        
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Elder Ray signals with regime alignment
        # Long: Bull Power > 0 (buying pressure) in trending up market
        # Short: Bear Power < 0 (selling pressure) in trending down market
        # We infer trend direction from recent price action vs EMA
        ema_now = ema13[i]
        ema_prev = ema13[i-1]
        short_term_trend = 1 if ema_now > ema_prev else -1
        
        enter_long = trending_regime and vol_confirm and (bull_power[i] > 0) and (short_term_trend > 0)
        enter_short = trending_regime and vol_confirm and (bear_power[i] < 0) and (short_term_trend < 0)
        
        # Final entry conditions
        if enter_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif enter_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #5051: 6h Elder Ray + 1d ADX Regime + Volume Confirmation
HYPOTHESIS: On 6h timeframe, Elder Ray (Bull Power/Bear Power) identifies momentum strength while 1d ADX regime filter (ADX>25) ensures trending markets. Volume > 1.5x average confirms participation. This combination works in both bull (Bull Power > 0 in uptrend) and bear (Bear Power < 0 in downtrend) markets by capturing institutional moves with proper regime alignment. Target: 12-37 trades/year on 6h to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5051_6h_elder_ray_1d_adx_regime_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for ADX regime
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: ADX(14) for regime filter ===
    if len(df_1d) >= 14:
        # Calculate True Range
        tr1 = df_1d['high'].values[1:] - df_1d['low'].values[1:]
        tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
        tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
        tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Calculate Directional Movement
        up_move = df_1d['high'].values[1:] - df_1d['high'].values[:-1]
        down_move = df_1d['low'].values[:-1] - df_1d['low'].values[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smooth TR and DM
        tr_14 = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
        plus_dm_14 = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
        minus_dm_14 = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Calculate DI and DX
        plus_di = 100 * plus_dm_14 / tr_14
        minus_di = 100 * minus_dm_14 / tr_14
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX
        adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Align to 6h timeframe
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    else:
        adx_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Elder Ray (Bull Power/Bear Power) ===
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(13, 20)  # EMA13, Volume MA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position when Elder Power reverses or ADX weakens ---
        if in_position:
            exit_long = (position_side > 0) and ((bull_power[i] <= 0) or (adx_aligned[i] < 20))
            exit_short = (position_side < 0) and ((bear_power[i] >= 0) or (adx_aligned[i] < 20))
            
            if exit_long or exit_short:
                in_position = False
                position_side = 0
                signals[i] = 0.0
            else:
                signals[i] = SIZE * position_side
            continue
        
        # --- New Position Entry Logic ---
        # Regime filter: ADX > 25 indicates trending market
        trending_regime = adx_aligned[i] > 25
        
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Elder Ray signals with regime alignment
        # Long: Bull Power > 0 (buying pressure) in trending up market
        # Short: Bear Power < 0 (selling pressure) in trending down market
        # We infer trend direction from recent price action vs EMA
        ema_now = ema13[i]
        ema_prev = ema13[i-1]
        short_term_trend = 1 if ema_now > ema_prev else -1
        
        enter_long = trending_regime and vol_confirm and (bull_power[i] > 0) and (short_term_trend > 0)
        enter_short = trending_regime and vol_confirm and (bear_power[i] < 0) and (short_term_trend < 0)
        
        # Final entry conditions
        if enter_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif enter_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals