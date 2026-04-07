#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with 1-day trend filter and volume confirmation
# Long when price breaks above Donchian(20) high, 1d close > 1d EMA100 (uptrend), and volume > 2x 4h average volume
# Short when price breaks below Donchian(20) low, 1d close < 1d EMA100 (downtrend), and volume > 2x 4h average volume
# Exit when Donchian reversal or trend changes
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-day EMA100 for trend filter and 4h volume average for confirmation
# Target: 80-150 total trades over 4 years (20-38/year)

name = "4h_donchian20_1d_ema100_vol_v1"
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
    
    # 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # 4h volume average for confirmation
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
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
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema100_1d_aligned[i]) or np.isnan(volume_ma[i]) or 
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
            # Exit: Donchian reversal or trend changes
            elif close[i] < donchian_low[i] or close[i] < ema100_1d_aligned[i]:
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
            # Exit: Donchian reversal or trend changes
            elif close[i] > donchian_high[i] or close[i] > ema100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Donchian breakout, trend alignment, and volume confirmation
            # Bullish breakout: price breaks above Donchian high
            bullish_break = close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1]
            # Bearish breakout: price breaks below Donchian low
            bearish_break = close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1]
            
            # Long: bullish breakout, 1d uptrend, volume spike
            if (bullish_break and
                close[i] > ema100_1d_aligned[i] and
                volume[i] > 2.0 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bearish breakout, 1d downtrend, volume spike
            elif (bearish_break and
                  close[i] < ema100_1d_aligned[i] and
                  volume[i] > 2.0 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot reversal with 1-day trend filter and volume confirmation
# Long when price touches Camarilla S3 level (strong support) in 1d uptrend with volume confirmation
# Short when price touches Camarilla R3 level (strong resistance) in 1d downtrend with volume confirmation
# Exit when price reaches opposite Camarilla level or trend changes
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-day EMA50 for trend filter and 4h volume average for confirmation
# Target: 70-140 total trades over 4 years (18-35/year)

name = "4h_camarilla_reversal_1d_ema50_vol_v1"
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
    
    # 4h Camarilla pivot levels (based on previous bar's OHLC)
    # Calculate for each bar using previous bar's data
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    s3 = pivot - (range_val * 1.1 / 6.0)  # Strong support
    r3 = pivot + (range_val * 1.1 / 6.0)  # Strong resistance
    s4 = pivot - (range_val * 1.1 / 4.0)  # Defend/sell
    r4 = pivot + (range_val * 1.1 / 4.0)  # Defend/buy
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h volume average for confirmation
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
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
        if (np.isnan(s3[i]) or np.isnan(r3[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ma[i]) or 
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
            # Exit: price reaches R3/S4 or trend changes
            elif close[i] > r3[i] or close[i] < ema50_1d_aligned[i]:
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
            # Exit: price reaches S3/R4 or trend changes
            elif close[i] < s3[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries at Camarilla S3/R3 levels with trend alignment and volume confirmation
            # Long: price touches S3 (strong support) in 1d uptrend with volume spike
            long_condition = (close[i] <= s3[i] * 1.001 and close[i] >= s3[i] * 0.999) and \
                           close[i] > ema50_1d_aligned[i] and \
                           volume[i] > 2.0 * volume_ma[i]
            # Short: price touches R3 (strong resistance) in 1d downtrend with volume spike
            short_condition = (close[i] <= r3[i] * 1.001 and close[i] >= r3[i] * 0.999) and \
                            close[i] < ema50_1d_aligned[i] and \
                            volume[i] > 2.0 * volume_ma[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour TRIX momentum with 1-day volume confirmation and ATR stoploss
# Long when TRIX crosses above zero line with rising 1d volume and price above 1d EMA50
# Short when TRIX crosses below zero line with rising 1d volume and price below 1d EMA50
# Exit when TRIX reverses or price crosses EMA50
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-day EMA50 for trend filter and 1d volume average for confirmation
# Target: 80-160 total trades over 4 years (20-40/year)

name = "4h_trix_momentum_1d_ema50_vol_v1"
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
    
    # 4h TRIX (triple exponential smoothing)
    # Calculate EMA1, EMA2, EMA3 then % change
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = (ema3.pct_change() * 100).values  # Percentage change
    
    # 1d data for trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # 4h volume average for additional confirmation
    volume_series = pd.Series(volume)
    volume_ma_4h = volume_series.rolling(window=20, min_periods=20).mean().values
    
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
        if (np.isnan(trix[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_ma_1d_aligned[i]) or np.isnan(volume_ma_4h[i]) or 
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
            # Exit: TRIX reverses or price crosses EMA50
            elif trix[i] < 0 or close[i] < ema50_1d_aligned[i]:
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
            # Exit: TRIX reverses or price crosses EMA50
            elif trix[i] > 0 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with TRIX zero-line cross, trend alignment, and volume confirmation
            # Bullish crossover: TRIX crosses above zero
            bullish_cross = trix[i] > 0 and trix[i-1] <= 0
            # Bearish crossover: TRIX crosses below zero
            bearish_cross = trix[i] < 0 and trix[i-1] >= 0
            
            # Volume confirmation: current 1d volume > 1.5x average
            volume_confirm = volume[i] > 1.5 * volume_ma_4h[i]
            
            # Long: bullish TRIX cross, 1d uptrend, volume confirmation
            if (bullish_cross and
                close[i] > ema50_1d_aligned[i] and
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bearish TRIX cross, 1d downtrend, volume confirmation
            elif (bearish_cross and
                  close[i] < ema50_1d_aligned[i] and
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams %R reversal with 1-day trend filter and volume confirmation
# Long when Williams %R crosses above -80 (oversold) in 1d uptrend with volume spike
# Short when Williams %R crosses below -20 (overbought) in 1d downtrend with volume spike
# Exit when Williams %R crosses opposite threshold or trend changes
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-day EMA100 for trend filter and 4h volume average for confirmation
# Target: 75-150 total trades over 4 years (19-38/year)

name = "4h_williamsr_reversal_1d_ema100_vol_v1"
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
    
    # 4h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # 4h volume average for confirmation
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
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
        if (np.isnan(williams_r[i]) or np.isnan(ema100_1d_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
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
            # Exit: Williams %R crosses above -20 or trend changes
            elif williams_r[i] > -20 or close[i] < ema100_1d_aligned[i]:
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
            # Exit: Williams %R crosses below -80 or trend changes
            elif williams_r[i] < -80 or close[i] > ema100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Williams %R reversal, trend alignment, and volume confirmation
            # Bullish reversal: Williams %R crosses above -80 (oversold)
            bullish_reversal = williams_r[i] > -80 and williams_r[i-1] <= -80
            # Bearish reversal: Williams %R crosses below -20 (overbought)
            bearish_reversal = williams_r[i] < -20 and williams_r[i-1] >= -20
            
            # Volume confirmation: current volume > 2.0x average
            volume_confirm = volume[i] > 2.0 * volume_ma[i]
            
            # Long: bullish reversal, 1d uptrend, volume confirmation
            if (bullish_reversal and
                close[i] > ema100_1d_aligned[i] and
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bearish reversal, 1d downtrend, volume confirmation
            elif (bearish_reversal and
                  close[i] < ema100_1d_aligned[i] and
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Elder Ray Index with 1-day trend filter and volume confirmation
# Long when Elder Ray Bull Power > 0 and Bear Power < 0 in 1d uptrend with volume spike
# Short when Elder Ray Bull Power < 0 and Bear Power > 0 in 1d downtrend with volume spike
# Exit when Elder Ray signals reverse or trend changes
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-day EMA50 for trend filter and 4h volume average for confirmation
# Target: 70-140 total trades over 4 years (18-35/year)

name = "4h_elder_ray_1d_ema50_vol_v1"
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
    
    # 4h Elder Ray Index (13-period EMA)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h volume average for confirmation
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
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
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ma[i]) or 
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
            # Exit: Elder Ray reverses or trend changes
            elif bull_power[i] <= 0 or bear_power[i] >= 0 or close[i] < ema50_1d_aligned[i]:
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
            # Exit: Elder Ray reverses or trend changes
            elif bull_power[i] >= 0 or bear_power[i] <= 0 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Elder Ray signals, trend alignment, and volume confirmation
            # Long: Bull Power > 0 and Bear Power < 0 (bullish) in 1d uptrend with volume spike
            long_condition = (bull_power[i] > 0 and bear_power[i] < 0) and \
                           close[i] > ema50_1d_aligned[i] and \
                           volume[i] > 2.0 * volume_ma[i]
            # Short: Bull Power < 0 and Bear Power > 0 (bearish) in 1d downtrend with volume spike
            short_condition = (bull_power[i] < 0 and bear_power[i] > 0) and \
                            close[i] < ema50_1d_aligned[i] and \
                            volume[i] > 2.0 * volume_ma[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Vortex Indicator with 1-day trend filter and volume confirmation
# Long when VI+ crosses above VI- and VI+ > 1.0 in 1d uptrend with volume spike
# Short when VI- crosses above VI+ and VI- > 1.0 in 1d downtrend with volume spike
# Exit when Vortex signals reverse or trend changes
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-day EMA100 for trend filter and 4h volume average for confirmation
# Target: 75-150 total trades over 4 years (19-38/year)

name = "4h_vortex_indicator_1d_ema100_vol_v1"
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
    
    # 4h Vortex Indicator (14-period)
    tr1 = np.abs(high - np.roll(low, 1))
    tr2 = np.abs(low - np.roll(high, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr = tr1 + tr2
    
    vm = np.abs(high - np.roll(low, 1))
    vm[0] = high[0] - low[0]
    
    # Sum over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_sum = pd.Series(vm).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    vi_plus = np.where(tr_sum != 0, vm_sum / tr_sum, 0)
    vi_minus = np.where(tr_sum != 0, pd.Series(np.abs(low