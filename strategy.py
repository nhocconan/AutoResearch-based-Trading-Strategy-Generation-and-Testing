#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian channel breakout with volume confirmation and ADX trend filter
# Long when price breaks above 20-period Donchian high with volume > 1.5x average and ADX > 25
# Short when price breaks below 20-period Donchian low with volume > 1.5x average and ADX > 25
# Exit when price returns to the middle of the Donchian channel or ADX < 20
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-day high/low for Donchian calculation to avoid look-ahead
# Target: 60-120 total trades over 4 years (15-30/year)

name = "12h_donchian20_1d_vol_adx_v1"
timeframe = "12h"
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
    
    # 1-day data for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels from previous 1d bar (shifted)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 20-period Donchian high and low (using previous day's data)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Align Donchian levels to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ADX(14) for trend strength on 12h
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_smooth = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI values
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR(14) for stoploss
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(adx[i]) or np.isnan(atr[i])):
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
            # Exit: price returns to middle of Donchian channel or ADX weakens (< 20)
            elif close[i] <= donch_mid_aligned[i] or adx[i] < 20:
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
            # Exit: price returns to middle of Donchian channel or ADX weakens (< 20)
            elif close[i] >= donch_mid_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price breaks Donchian channel with volume spike and strong trend
            # Volume spike: > 1.5x average volume
            volume_spike = volume[i] > 1.5 * volume_ma[i]
            # Strong trend: ADX > 25
            strong_trend = adx[i] > 25
            
            # Long: price breaks above Donchian high, volume spike, strong trend
            if (close[i] > donch_high_aligned[i] and 
                volume_spike and strong_trend):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low, volume spike, strong trend
            elif (close[i] < donch_low_aligned[i] and 
                  volume_spike and strong_trend):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot levels from daily chart with volume spike and ADX trend filter
# Long when price touches S1 support with volume > 1.5x average and ADX > 25 (trending)
# Short when price touches R1 resistance with volume > 1.5x average and ADX > 25 (trending)
# Exit when price reaches the pivot point (PP) or ADX < 20 (range)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-day OHLC for Camarilla calculation to avoid look-ahead
# Target: 60-120 total trades over 4 years (15-30/year)

name = "12h_camarilla_1d_vol_adx_v1"
timeframe = "12h"
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
    
    # 1-day data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Camarilla levels: based on previous day's range
    # R4 = close + (high - low) * 1.500
    # R3 = close + (high - low) * 1.250
    # R2 = close + (high - low) * 1.166
    # R1 = close + (high - low) * 1.083
    # S1 = close - (high - low) * 1.083
    # S2 = close - (high - low) * 1.166
    # S3 = close - (high - low) * 1.250
    # S4 = close - (high - low) * 1.500
    r1_1d = close_1d + (high_1d - low_1d) * 1.083
    s1_1d = close_1d - (high_1d - low_1d) * 1.083
    
    # Align Camarilla levels to 12h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ADX(14) for trend strength on 12h
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_smooth = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI values
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR(14) for stoploss
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(pp_1d_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(adx[i]) or np.isnan(atr[i])):
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
            # Exit: price reaches PP or ADX weakens (< 20)
            elif close[i] >= pp_1d_aligned[i] or adx[i] < 20:
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
            # Exit: price reaches PP or ADX weakens (< 20)
            elif close[i] <= pp_1d_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price touches S1/R1 with volume spike and strong trend
            # Volume spike: > 1.5x average volume
            volume_spike = volume[i] > 1.5 * volume_ma[i]
            # Strong trend: ADX > 25
            strong_trend = adx[i] > 25
            
            # Long: price touches S1 support, volume spike, strong trend
            if (close[i] <= s1_1d_aligned[i] * 1.001 and  # Allow small tolerance for touch
                volume_spike and strong_trend):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price touches R1 resistance, volume spike, strong trend
            elif (close[i] >= r1_1d_aligned[i] * 0.999 and  # Allow small tolerance for touch
                  volume_spike and strong_trend):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Triple Exponential Moving Average (TEMA) crossover with volume confirmation and ADX trend filter
# Long when TEMA(9) crosses above TEMA(21) with volume > 1.5x average and ADX > 25
# Short when TEMA(9) crosses below TEMA(21) with volume > 1.5x average and ADX > 25
# Exit when TEMA(9) crosses back through TEMA(21) or ADX < 20
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-day TEMA for trend context to avoid whipsaws
# Target: 60-120 total trades over 4 years (15-30/year)

name = "12h_tema_cross_1d_vol_adx_v1"
timeframe = "12h"
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
    
    # 1-day data for TEMA trend context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate TEMA on 1d for trend filter
    close_1d = df_1d['close'].values
    ema1_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema2_1d = pd.Series(ema1_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema3_1d = pd.Series(ema2_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    tema_1d = 3 * ema1_1d - 3 * ema2_1d + ema3_1d
    tema_1d_aligned = align_htf_to_ltf(prices, df_1d, tema_1d)
    
    # Calculate TEMA on 12h for entry signals
    ema1 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    tema_fast = 3 * ema1 - 3 * ema2 + ema3
    
    ema1_slow = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema2_slow = pd.Series(ema1_slow).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema3_slow = pd.Series(ema2_slow).ewm(span=21, adjust=False, min_periods=21).mean().values
    tema_slow = 3 * ema1_slow - 3 * ema2_slow + ema3_slow
    
    # TEMA crossover signals
    tema_cross_up = (tema_fast > tema_slow) & (np.roll(tema_fast, 1) <= np.roll(tema_slow, 1))
    tema_cross_down = (tema_fast < tema_slow) & (np.roll(tema_fast, 1) >= np.roll(tema_slow, 1))
    tema_cross_up[0] = False
    tema_cross_down[0] = False
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ADX(14) for trend strength on 12h
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_smooth = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI values
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR(14) for stoploss
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(tema_1d_aligned[i]) or np.isnan(tema_fast[i]) or 
            np.isnan(tema_slow[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(adx[i]) or np.isnan(atr[i])):
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
            # Exit: TEMA cross down or ADX weakens (< 20)
            elif tema_cross_down[i] or adx[i] < 20:
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
            # Exit: TEMA cross up or ADX weakens (< 20)
            elif tema_cross_up[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: TEMA crossover with volume spike and strong trend
            # Volume spike: > 1.5x average volume
            volume_spike = volume[i] > 1.5 * volume_ma[i]
            # Strong trend: ADX > 25
            strong_trend = adx[i] > 25
            # 1d TEMA trend filter: only long if 1d TEMA rising, short if falling
            tema_1d_rising = tema_1d_aligned[i] > tema_1d_aligned[i-1] if i > 0 else False
            tema_1d_falling = tema_1d_aligned[i] < tema_1d_aligned[i-1] if i > 0 else False
            
            # Long: TEMA(9) crosses above TEMA(21), volume spike, strong trend, 1d TEMA rising
            if (tema_cross_up[i] and volume_spike and strong_trend and tema_1d_rising):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: TEMA(9) crosses below TEMA(21), volume spike, strong trend, 1d TEMA falling
            elif (tema_cross_down[i] and volume_spike and strong_trend and tema_1d_falling):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Relative Strength Index (RSI) with Bollinger Bands and volume confirmation
# Long when RSI < 30 and price touches lower Bollinger Band with volume > 1.5x average
# Short when RSI > 70 and price touches upper Bollinger Band with volume > 1.5x average
# Exit when price reaches the middle Bollinger Band or RSI returns to 50
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-day trend filter (price > EMA50 for longs, price < EMA50 for shorts)
# Target: 60-120 total trades over 4 years (15-30/year)

name = "12h_rsi_bb_1d_vol_trend_v1"
timeframe = "12h"
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
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi[0] = 50  # Initialize first value
    
    # Bollinger Bands (20, 2)
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ADX(14) for trend strength confirmation (optional)
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_smooth = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI values
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR(14) for stoploss
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(bb_middle[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(atr[i])):
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
            # Exit: price reaches middle BB or RSI returns to 50
            elif close[i] >= bb_middle[i] or rsi[i] >= 50:
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
            # Exit: price reaches middle BB or RSI returns to 50
            elif close[i] <= bb_middle[i] or rsi[i] <= 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: RSI extreme + BB touch + volume spike + 1d trend filter
            # Volume spike: > 1.5x average volume
            volume_spike = volume[i] > 1.5 * volume_ma[i]
            
            # Long: RSI < 30, price touches lower BB, volume spike, price > 1d EMA50
            if (rsi[i] < 30 and 
                close[i] <= bb_lower[i] * 1.001 and  # Allow small tolerance