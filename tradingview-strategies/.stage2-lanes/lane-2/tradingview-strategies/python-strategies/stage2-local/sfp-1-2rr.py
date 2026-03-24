import numpy as np
import pandas as pd

name = "SFP 1:2RR"
timeframe = "5m"

def generate_signals(prices):
    """
    Generates position intent signals (1=Long, -1=Short, 0=Flat) based on
    SFP 1:2RR logic (HTF Swings/Mitigation -> LTF Displacement/FVG).
    
    Args:
        prices (pd.DataFrame): Must contain columns ['open_time', 'open', 'high', 'low', 'close', 'volume'].
                               'open_time' should be datetime-like.
    
    Returns:
        np.ndarray: Integer array of length len(prices) with values 1, -1, or 0.
    """
    if prices.empty:
        return np.array([], dtype=np.int8)
    
    df = prices.copy()
    if 'open_time' in df.columns:
        df = df.set_index('open_time')
    
    n = len(df)
    if n < 10:
        return np.zeros(n, dtype=np.int8)
    
    # HTF (1h) Resampling
    htf = df.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).shift(1).dropna()
    
    if htf.empty or len(htf) < 3:
        return np.zeros(n, dtype=np.int8)
    
    # Calculate HTF Swings (Pivot High/Low with left=1, right=1)
    htf = htf.copy()
    htf['ph'] = False
    htf['pl'] = False
    htf_vals = htf.values
    for i in range(1, len(htf) - 1):
        if htf_vals[i, 1] > htf_vals[i-1, 1] and htf_vals[i, 1] > htf_vals[i+1, 1]:
            htf.iloc[i, htf.columns.get_loc('ph')] = True
        if htf_vals[i, 2] < htf_vals[i-1, 2] and htf_vals[i, 2] < htf_vals[i+1, 2]:
            htf.iloc[i, htf.columns.get_loc('pl')] = True
    
    # Track Last Swing High/Low
    last_sh = np.nan
    last_sl = np.nan
    htf['last_swing_high'] = np.nan
    htf['last_swing_low'] = np.nan
    
    for i in range(len(htf)):
        if htf['ph'].iloc[i]:
            last_sh = float(htf['high'].iloc[i])
        if htf['pl'].iloc[i]:
            last_sl = float(htf['low'].iloc[i])
        htf.iloc[i, htf.columns.get_loc('last_swing_high')] = last_sh
        htf.iloc[i, htf.columns.get_loc('last_swing_low')] = last_sl
    
    # HTF Mitigation (SFP)
    htf['bear_mit'] = False
    htf['bull_mit'] = False
    for i in range(len(htf)):
        lsh = htf['last_swing_high'].iloc[i]
        lsl = htf['last_swing_low'].iloc[i]
        if not np.isnan(lsh) and htf['high'].iloc[i] > lsh and htf['close'].iloc[i] < lsh:
            htf.iloc[i, htf.columns.get_loc('bear_mit')] = True
        if not np.isnan(lsl) and htf['low'].iloc[i] < lsl and htf['close'].iloc[i] > lsl:
            htf.iloc[i, htf.columns.get_loc('bull_mit')] = True
    
    htf['htf_dir'] = 0
    htf.loc[htf['bull_mit'], 'htf_dir'] = 1
    htf.loc[htf['bear_mit'], 'htf_dir'] = -1
    
    htf['sweep_hi'] = np.nan
    htf['sweep_lo'] = np.nan
    for i in range(len(htf)):
        if htf['bear_mit'].iloc[i]:
            htf.iloc[i, htf.columns.get_loc('sweep_hi')] = htf['high'].iloc[i]
        if htf['bull_mit'].iloc[i]:
            htf.iloc[i, htf.columns.get_loc('sweep_lo')] = htf['low'].iloc[i]
    
    # Map HTF data back to 5m (forward fill)
    htf_aligned = htf.reindex(df.index, method='ffill').fillna(0)
    
    df['htf_dir'] = htf_aligned['htf_dir'].values
    df['sweep_hi'] = htf_aligned['sweep_hi'].values
    df['sweep_lo'] = htf_aligned['sweep_lo'].values
    
    # LTF (5m) Indicators
    close_shift = df['close'].shift(1).fillna(df['close'])
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - close_shift),
                               abs(df['low'] - close_shift)))
    df['atr'] = tr.rolling(window=14).mean().fillna(0)
    
    df['body'] = abs(df['close'] - df['open'])
    df['is_bull_disp'] = (df['close'] > df['open']) & (df['body'] >= df['atr'] * 0.8)
    df['is_bear_disp'] = (df['close'] < df['open']) & (df['body'] >= df['atr'] * 0.8)
    
    high_shift2 = df['high'].shift(2).fillna(df['high'])
    low_shift2 = df['low'].shift(2).fillna(df['low'])
    df['bull_fvg'] = df['low'] > high_shift2
    df['bear_fvg'] = df['high'] < low_shift2
    
    # State Machine Loop
    signals = np.zeros(n, dtype=np.int8)
    htf_times = htf_aligned.index.values
    
    active_dir = 0
    active_sl = np.nan
    mit_bar_index = -1
    last_mit_stamp = None
    traded_this_mit = False
    disp_seen = False
    fvg_top = np.nan
    fvg_bot = np.nan
    fvg_dir = 0
    in_trade = False
    trade_dir = 0
    prev_long_setup = False
    prev_short_setup = False
    
    for i in range(n):
        row = df.iloc[i]
        htf_time = htf_times[i] if i < len(htf_times) else None
        
        htf_dir_val = row['htf_dir']
        if np.isnan(htf_dir_val):
            htf_dir_val = 0
        htf_dir_val = int(htf_dir_val)
        
        new_mit_confirmed = False
        
        if htf_dir_val != 0:
            if last_mit_stamp is None or (htf_time is not None and htf_time != last_mit_stamp):
                if htf_dir_val != active_dir:
                    new_mit_confirmed = True
        
        if new_mit_confirmed:
            active_dir = htf_dir_val
            mit_bar_index = i
            last_mit_stamp = htf_time
            traded_this_mit = False
            disp_seen = False
            fvg_top = np.nan
            fvg_bot = np.nan
            fvg_dir = 0
            in_trade = False
            trade_dir = 0
            
            if active_dir == -1:
                sweep_hi_val = row['sweep_hi']
                if not np.isnan(sweep_hi_val):
                    active_sl = sweep_hi_val
            elif active_dir == 1:
                sweep_lo_val = row['sweep_lo']
                if not np.isnan(sweep_lo_val):
                    active_sl = sweep_lo_val
        
        disp_ok = False
        if active_dir == 1 and row['is_bull_disp']:
            disp_ok = True
        elif active_dir == -1 and row['is_bear_disp']:
            disp_ok = True
            
        if active_dir != 0 and disp_ok:
            disp_seen = True
            
        after_mit = (mit_bar_index != -1) and (i > mit_bar_index)
        
        if after_mit and disp_seen:
            if active_dir == 1 and row['bull_fvg']:
                fvg_top = float(row['low'])
                if i >= 2:
                    fvg_bot = float(df['high'].iloc[i-2])
                fvg_dir = 1
            elif active_dir == -1 and row['bear_fvg']:
                fvg_bot = float(row['high'])
                if i >= 2:
                    fvg_top = float(df['low'].iloc[i-2])
                fvg_dir = -1
        
        in_fvg = False
        if fvg_dir != 0 and not np.isnan(fvg_top) and not np.isnan(fvg_bot):
            top = max(fvg_top, fvg_bot)
            bot = min(fvg_top, fvg_bot)
            if row['low'] <= top and row['high'] >= bot:
                in_fvg = True
        
        can_trade = (active_dir != 0) and (not traded_this_mit) and (not in_trade)
        
        long_setup = can_trade and (fvg_dir == 1) and in_fvg
        short_setup = can_trade and (fvg_dir == -1) and in_fvg
        
        # Check Entry (from previous bar setup)
        if prev_long_setup and active_dir == 1:
            if not np.isnan(active_sl):
                risk = row['open'] - active_sl
                if risk > 0:
                    in_trade = True
                    trade_dir = 1
                    traded_this_mit = True
        elif prev_short_setup and active_dir == -1:
            if not np.isnan(active_sl):
                risk = active_sl - row['open']
                if risk > 0:
                    in_trade = True
                    trade_dir = -1
                    traded_this_mit = True
        
        # Set Signal
        if in_trade:
            signals[i] = int(trade_dir)
        else:
            signals[i] = 0
            
        prev_long_setup = long_setup
        prev_short_setup = short_setup
        
    return signals
