#!/usr/bin/env python3
"""
Experiment #1539: 6h Elder Ray + Regime Filter (Bull/Bear Power + ADX)
HYPOTHESIS: Elder Ray Index (Bull Power = High - EMA13, Bear Power = EMA13 - Low) captures buying/selling pressure. Combined with ADX regime filter (ADX>25 for trending, ADX<20 for ranging) and 12h trend alignment, this strategy enters on pullbacks in the trend direction. Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets. Target: 75-150 total trades over 4 years (19-37/year) by requiring confluence of Elder Ray extreme, ADX regime, and 12h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1539_6h_elder_ray_regime_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = EMA13 - Low
    bear_power = ema13 - low
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: ADX(14) for regime filter ===
    # +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Smoothed +DM, -DM, TR
    tr_ma = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_dm_ma = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    minus_dm_ma = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_ma / tr_ma
    minus_di = 100 * minus_dm_ma / tr_ma
    
    # DX and ADX
    dx = np.zeros(n)
    dx[14:] = 100 * np.abs(plus_di[14:] - minus_di[14:]) / (plus_di[14:] + minus_di[14:])
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === HTF: 12h data for trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    trend_12h = np.where(close_12h > ema_12h, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 30  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr[i]) or np.isnan(adx[i]) or np.isnan(trend_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Regime filter: ADX > 25 for trending market
        trending = adx[i] > 25
        
        # Elder Ray signals: look for extremes in trending markets
        # Bull Power making new high suggests strong buying pressure
        # Bear Power making new low suggests strong selling pressure
        if trending:
            # Check for Bull Power expansion (making new high)
            bull_expanding = bull_power[i] > bull_power[i-1] and bull_power[i] > 0
            # Check for Bear Power expansion (making new low - more negative)
            bear_expanding = bear_power[i] > bear_power[i-1] and bear_power[i] > 0
            
            # Entry logic: pullback in trend direction with Elder Ray confirmation
            if trend_12h_aligned[i] > 0:  # 12h uptrend
                # Look for bear power expansion (selling pressure) as pullback to buy
                if bear_expanding and bear_power[i] > 0:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
            elif trend_12h_aligned[i] < 0:  # 12h downtrend
                # Look for bull power expansion (buying pressure) as rally to sell
                if bull_expanding and bull_power[i] > 0:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #1539: 6h Elder Ray + Regime Filter (Bull/Bear Power + ADX)
HYPOTHESIS: Elder Ray Index (Bull Power = High - EMA13, Bear Power = EMA13 - Low) captures buying/selling pressure. Combined with ADX regime filter (ADX>25 for trending, ADX<20 for ranging) and 12h trend alignment, this strategy enters on pullbacks in the trend direction. Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets. Target: 75-150 total trades over 4 years (19-37/year) by requiring confluence of Elder Ray extreme, ADX regime, and 12h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1539_6h_elder_ray_regime_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = EMA13 - Low
    bear_power = ema13 - low
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: ADX(14) for regime filter ===
    # +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Smoothed +DM, -DM, TR
    tr_ma = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_dm_ma = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    minus_dm_ma = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_ma / tr_ma
    minus_di = 100 * minus_dm_ma / tr_ma
    
    # DX and ADX
    dx = np.zeros(n)
    dx[14:] = 100 * np.abs(plus_di[14:] - minus_di[14:]) / (plus_di[14:] + minus_di[14:])
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === HTF: 12h data for trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    trend_12h = np.where(close_12h > ema_12h, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 30  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr[i]) or np.isnan(adx[i]) or np.isnan(trend_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Regime filter: ADX > 25 for trending market
        trending = adx[i] > 25
        
        # Elder Ray signals: look for extremes in trending markets
        # Bull Power making new high suggests strong buying pressure
        # Bear Power making new low suggests strong selling pressure
        if trending:
            # Check for Bull Power expansion (making new high)
            bull_expanding = bull_power[i] > bull_power[i-1] and bull_power[i] > 0
            # Check for Bear Power expansion (making new low - more negative)
            bear_expanding = bear_power[i] > bear_power[i-1] and bear_power[i] > 0
            
            # Entry logic: pullback in trend direction with Elder Ray confirmation
            if trend_12h_aligned[i] > 0:  # 12h uptrend
                # Look for bear power expansion (selling pressure) as pullback to buy
                if bear_expanding and bear_power[i] > 0:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
            elif trend_12h_aligned[i] < 0:  # 12h downtrend
                # Look for bull power expansion (buying pressure) as rally to sell
                if bull_expanding and bull_power[i] > 0:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
    
    return signals

</think>