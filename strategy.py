#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour strategy using 4-hour Supertrend for direction and 1-hour EMA crossover for entry timing
# Long when 4h Supertrend is bullish (price above Supertrend line) and 1h EMA(8) crosses above EMA(21)
# Short when 4h Supertrend is bearish (price below Supertrend line) and 1h EMA(8) crosses below EMA(21)
# Exit when EMA(8) crosses back in opposite direction
# Stoploss at 2.0 * ATR(14) from entry price
# Position size: 0.20 (20% of capital)
# Uses Supertrend for trend direction (4h) and EMA crossover for precise entries (1h)
# Target: 60-150 total trades over 4 years (15-37/year)

name = "1h_supertrend4h_ema8_21_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4-hour data for Supertrend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate Supertrend on 4h data
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR for 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = pd.Series(tr_4h).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate upper and lower bands
    hl2 = (high_4h + low_4h) / 2
    upper_band = hl2 + (multiplier * atr_4h)
    lower_band = hl2 - (multiplier * atr_4h)
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close_4h)
    direction = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_4h)):
        if close_4h[i] > upper_band[i-1]:
            direction[i] = 1
        elif close_4h[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Align Supertrend direction to 1h timeframe
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_4h, direction)
    
    # 1-hour indicators for entry timing
    # EMA(8) and EMA(21)
    ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(21, n):
        # Skip if required data not available
        if (np.isnan(ema8[i]) or np.isnan(ema21[i]) or 
            np.isnan(supertrend_direction_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: EMA(8) crosses below EMA(21)
            elif ema8[i] < ema21[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: EMA(8) crosses above EMA(21)
            elif ema8[i] > ema21[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: EMA crossover with 4h Supertrend direction filter
            # Long: EMA(8) crosses above EMA(21) and 4h Supertrend is bullish
            if ema8[i] > ema21[i] and ema8[i-1] <= ema21[i-1] and supertrend_direction_aligned[i] == 1:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: EMA(8) crosses below EMA(21) and 4h Supertrend is bearish
            elif ema8[i] < ema21[i] and ema8[i-1] >= ema21[i-1] and supertrend_direction_aligned[i] == -1:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour strategy using 1-day ADX for trend strength and 1-hour RSI for mean reversion entries
# Long when 1d ADX > 25 (trending) and 1h RSI(14) < 30 (oversold) in uptrend
# Short when 1d ADX > 25 (trending) and 1h RSI(14) > 70 (overbought) in downtrend
# Exit when RSI returns to neutral zone (40-60)
# Stoploss at 2.5 * ATR(14) from entry price
# Position size: 0.20 (20% of capital)
# Uses daily ADX to filter only trending markets and hourly RSI for precise mean reversion entries
# Target: 60-150 total trades over 4 years (15-37/year)

name = "1h_adx1d_rsi14_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-day data for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 1d data
    adx_period = 14
    
    # Calculate True Range for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_smooth = pd.Series(tr_1d).ewm(alpha=1/adx_period, adjust=False, min_periods=adx_period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/adx_period, adjust=False, min_periods=adx_period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/adx_period, adjust=False, min_periods=adx_period).mean().values
    
    # Calculate DI+ and DI-
    plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
    minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/adx_period, adjust=False, min_periods=adx_period).mean().values
    
    # Align ADX to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1-hour indicators
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI returns to neutral (above 40)
            elif rsi[i] > 40:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI returns to neutral (below 60)
            elif rsi[i] < 60:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: RSI extreme with ADX trend filter
            # Long: ADX > 25 (trending) and RSI < 30 (oversold)
            if adx_aligned[i] > 25 and rsi[i] < 30:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: ADX > 25 (trending) and RSI > 70 (overbought)
            elif adx_aligned[i] > 25 and rsi[i] > 70:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour strategy using 4-hour volume-weighted average price (VWAP) deviation and 1-hour momentum
# Long when price is below 4h VWAP by 2% and 1h ROC(10) > 0 (bullish momentum)
# Short when price is above 4h VWAP by 2% and 1h ROC(10) < 0 (bearish momentum)
# Exit when price returns to within 0.5% of 4h VWAP
# Stoploss at 2.0 * ATR(14) from entry price
# Position size: 0.20 (20% of capital)
# Uses 4h VWAP as dynamic support/resistance and hourly momentum for entry timing
# Target: 60-150 total trades over 4 years (15-37/year)

name = "1h_vwap4h_roc10_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4-hour data for VWAP
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate VWAP on 4h data
    typical_price_4h = (df_4h['high'].values + df_4h['low'].values + df_4h['close'].values) / 3
    vwap_numerator = np.cumsum(typical_price_4h * df_4h['volume'].values)
    vwap_denominator = np.cumsum(df_4h['volume'].values)
    vwap_4h = vwap_numerator / (vwap_denominator + 1e-10)
    
    # Align VWAP to 1h timeframe
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    
    # 1-hour indicators
    # ROC(10) for momentum
    roc = np.zeros_like(close)
    roc[10:] = (close[10:] - close[:-10]) / close[:-10] * 100
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(10, n):
        # Skip if required data not available
        if (np.isnan(vwap_4h_aligned[i]) or np.isnan(roc[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to within 0.5% of VWAP
            elif abs(close[i] - vwap_4h_aligned[i]) / vwap_4h_aligned[i] < 0.005:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to within 0.5% of VWAP
            elif abs(close[i] - vwap_4h_aligned[i]) / vwap_4h_aligned[i] < 0.005:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: VWAP deviation with momentum confirmation
            # Long: price below VWAP by 2% and positive momentum (ROC > 0)
            if close[i] < vwap_4h_aligned[i] * 0.98 and roc[i] > 0:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price above VWAP by 2% and negative momentum (ROC < 0)
            elif close[i] > vwap_4h_aligned[i] * 1.02 and roc[i] < 0:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour strategy using 1-day Donchian channels for breakout direction and 1-hour volume surge for entry timing
# Long when price breaks above 1d Donchian(20) high and 1h volume > 2.0x 20-period 1h volume average
# Short when price breaks below 1d Donchian(20) low and 1h volume > 2.0x 20-period 1h volume average
# Exit when price returns to the midpoint of the Donchian channel
# Stoploss at 2.5 * ATR(14) from entry price
# Position size: 0.20 (20% of capital)
# Uses daily Donchian for trend structure and hourly volume surge for precise breakout entries
# Target: 60-150 total trades over 4 years (15-37/year)

name = "1h_donchian1d_volsurge_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on 1d data
    highest_high = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 1h timeframe
    highest_high_aligned = align_htf_to_ltf(prices, df_1d, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_1d, lowest_low)
    
    # 1-hour indicators
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to midpoint of Donchian channel
            elif close[i] > (highest_high_aligned[i] + lowest_low_aligned[i]) / 2:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to midpoint of Donchian channel
            elif close[i] < (highest_high_aligned[i] + lowest_low_aligned[i]) / 2:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: Donchian breakout with volume surge confirmation
            # Volume surge: volume > 2.0x 20-period volume average
            volume_surge = volume[i] > 2.0 * volume_ma[i]
            
            # Long: price breaks above 1d Donchian high + volume surge
            if close[i] > highest_high_aligned[i] and volume_surge:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price breaks below 1d Donchian low + volume surge
            elif close[i] < lowest_low_aligned[i] and volume_surge:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour strategy using 4-hour MACD histogram for trend direction and 1-hour Stochastic oscillator for mean reversion entries
# Long when 4h MACD histogram > 0 (bullish momentum) and 1h Stochastic %K < 20 (oversold)
# Short when 4h MACD histogram < 0 (bearish momentum) and 1h Stochastic %K > 80 (overbought)
# Exit when Stochastic %K crosses 50 in opposite direction
# Stoploss at 2.0 * ATR(14) from entry price
# Position size: 0.20 (20% of capital)
# Uses 4h MACD for trend filter and 1h Stochastic for precise mean reversion entries
# Target: 60-150 total trades over 4 years (15-37/year)

name = "1h_macd4h_stoch14_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4-hour data for MACD
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate MACD on 4h data
    fast_period = 12
    slow_period = 26
    signal_period = 9
    
    close_4h = df_4h['close'].values
    
    # Calculate EMAs
    ema_fast = pd.Series(close_4h).ewm(span=fast_period, adjust=False, min_periods=fast_period).mean().values
    ema_slow = pd.Series(close_4h).ewm(span=slow_period, adjust=False, min_periods=slow_period).mean().values
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = pd.Series(macd_line).ewm(span=signal_period, adjust=False, min_periods=signal_period).mean().values
    
    # Calculate MACD histogram
    macd_hist = macd_line - signal_line
    
    # Align MACD histogram to 1h timeframe
    macd_hist_aligned = align_htf_to_ltf(prices, df_4h, macd_hist)
    
    # 1-hour indicators
    # Stochastic oscillator (14,3,3)
    stoch_period = 14
    k_smooth = 3
    d_smooth = 3
    
    lowest_low = pd.Series(low).rolling(window=stoch_period, min_periods=stoch_period).min().values
    highest_high = pd.Series(high).rolling(window=stoch_period, min_periods=stoch_period).max().values
    
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl == 0, 1, range_hl)
    
    stoch_k = 100 * (close - lowest_low) / range_hl
    
    # Smooth %K to get %D
    stoch_d = pd.Series(stoch_k).ewm(alpha=1/k_smooth, adjust=False, min_periods=k_smooth).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(stoch_period, n):
        # Skip if required data not available
        if (np.isnan(macd_hist_aligned[i]) or np.isnan(stoch_k[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Stochastic %K crosses below 50
            elif stoch_k[i] < 50 and stoch_k[i-1] >= 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Stochastic %K crosses above 50
            elif stoch_k[i] > 50 and stoch_k[i-1] <= 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: MACD histogram direction with Stochastic extreme
            # Long: MACD histogram > 0 (bullish) and Stochastic %K < 20 (oversold)
            if macd_hist_aligned[i] > 0 and stoch_k[i] < 20:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: MACD histogram < 0 (bearish) and Stochastic %K > 80 (overbought)
            elif macd_hist_aligned[i] < 0 and stoch_k[i] > 80:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour strategy using 1-day Bollinger Bands for volatility context and 1-hour RSI divergence for mean reversion entries
# Long when price touches 1d Bollinger lower band and 1h RSI shows bullish divergence (higher low in RSI while price makes lower low)
# Short when price touches 1d Bollinger upper band and 1h RSI shows bearish divergence (lower high in RSI while price makes higher high)
# Exit when price crosses the 1d Bollinger middle band (20-period SMA)
# Stoploss at 2.0 * ATR(14) from entry price
# Position size: 0.20 (2