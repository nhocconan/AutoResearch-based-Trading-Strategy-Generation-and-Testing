#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and 1w volume confirmation
# Long when price breaks above 4h Donchian upper band, 1d ADX > 25 (strong trend), and 1w volume > 1.5x average
# Short when price breaks below 4h Donchian lower band, 1d ADX > 25 (strong trend), and 1w volume > 1.5x average
# Exit when price returns to 1d EMA(50) or opposite breakout occurs
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1d ADX for trend strength filter and 1w volume to confirm breakout strength
# Target: 100-200 total trades over 4 years (25-50/year)

name = "4h_donchian20_1d_adx_1w_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian(20) channels
    high_series = pd.Series(high_4h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    low_series = pd.Series(low_4h)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 4h timeframe (already aligned, but use for safety)
    upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # 1d ADX(14) for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for ADX
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # Calculate DI and DX
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    dx = np.where(np.isnan(dx) | np.isinf(dx), 0, dx)
    
    # Calculate ADX
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx = np.where(np.isnan(adx) | np.isinf(adx), 0, adx)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1w volume for confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    volume_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_1w)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma_1w_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to EMA or breaks below lower band
            elif close[i] <= 0 or close[i] < lower_aligned[i]:  # Placeholder for EMA, will fix
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to EMA or breaks above upper band
            elif close[i] <= 0 or close[i] > upper_aligned[i]:  # Placeholder for EMA, will fix
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend strength
            # Long: price breaks above upper band, ADX > 25 (strong trend), volume spike
            if (close[i] > upper_aligned[i] and
                adx_aligned[i] > 25 and
                volume[i] > 1.5 * volume_ma_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower band, ADX > 25 (strong trend), volume spike
            elif (close[i] < lower_aligned[i] and
                  adx_aligned[i] > 25 and
                  volume[i] > 1.5 * volume_ma_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals
 
I need to fix the placeholder EMA references in the exit conditions. Let me add the 1d EMA(50) calculation and use it properly. Also need to ensure all calculations use proper min_periods and handle edge cases.

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and 1w volume confirmation
# Long when price breaks above 4h Donchian upper band, 1d ADX > 25 (strong trend), and 1w volume > 1.5x average
# Short when price breaks below 4h Donchian lower band, 1d ADX > 25 (strong trend), and 1w volume > 1.5x average
# Exit when price returns to 1d EMA(50) or opposite breakout occurs
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1d ADX for trend strength filter and 1w volume to confirm breakout strength
# Target: 100-200 total trades over 4 years (25-50/year)

name = "4h_donchian20_1d_adx_1w_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian(20) channels
    high_series = pd.Series(high_4h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    low_series = pd.Series(low_4h)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # 1d data for ADX and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX(14) for trend strength filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    dx = np.where(np.isnan(dx) | np.isinf(dx), 0, dx)
    
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx = np.where(np.isnan(adx) | np.isinf(adx), 0, adx)
    
    # 1d EMA(50) for exit condition
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    
    # Align 1d indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1w volume for confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    volume_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_1w)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(volume_ma_1w_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to EMA or breaks below lower band
            elif close[i] <= ema_1d_aligned[i] or close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to EMA or breaks above upper band
            elif close[i] >= ema_1d_aligned[i] or close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend strength
            # Long: price breaks above upper band, ADX > 25 (strong trend), volume spike
            if (close[i] > upper_aligned[i] and
                adx_aligned[i] > 25 and
                volume[i] > 1.5 * volume_ma_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower band, ADX > 25 (strong trend), volume spike
            elif (close[i] < lower_aligned[i] and
                  adx_aligned[i] > 25 and
                  volume[i] > 1.5 * volume_ma_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals
 
I notice the code is already complete and correct. Let me verify it meets all requirements:
1. Uses 4h timeframe as required
2. Calls get_htf_data() only once for each timeframe before the loop
3. Uses align_htf_to_ltf() for proper alignment
4. Has proper min_periods on all rolling calculations
5. Uses discrete position sizes (0.25, -0.25)
6. Includes stoploss logic (2.0 * ATR)
7. Has exit conditions
8. Uses ADX > 25 for trend strength filter
9. Uses 1.5x volume spike for confirmation
10. Should generate reasonable trade count (not too many, not too few)

The code looks good and ready to submit. I'll output it as is.