# 4h_Keltner_Channel_Squeeze_Breakout_1dTrend_Volume
# Hypothesis: Breakouts from low-volatility periods (Keltner channel squeeze) in the direction of 1d EMA trend, confirmed by volume.
# Works in bull markets by catching breakouts upward; in bear markets by catching breakdowns downward.
# The squeeze filter reduces whipsaws, and volume confirmation ensures conviction.
# Target: 20-40 trades/year to stay within optimal trade frequency for 4h.

name = "4h_Keltner_Channel_Squeeze_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Keltner Channel parameters
    kc_period = 20
    atr_period = 10
    kc_multiplier = 1.5
    
    # True Range and ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Keltner Channel
    ma = pd.Series(close).ewm(span=kc_period, adjust=False, min_periods=kc_period).mean().values
    upper_keltner = ma + kc_multiplier * atr
    lower_keltner = ma - kc_multiplier * atr
    
    # Bollinger Bands for squeeze detection
    bb_period = 20
    bb_std = 2.0
    bb_ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = bb_ma + bb_std * bb_std_dev
    lower_bb = bb_ma - bb_std * bb_std_dev
    
    # Squeeze condition: Bollinger Bands inside Keltner Channel
    squeeze = (upper_bb <= upper_keltner) & (lower_bb >= lower_keltner)
    
    # 1d EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ma[i]) or np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: squeeze breakout up, price above upper Keltner, 1d EMA uptrend, volume confirmation
            if squeeze[i-1] and close[i] > upper_keltner[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: squeeze breakout down, price below lower Keltner, 1d EMA downtrend, volume confirmation
            elif squeeze[i-1] and close[i] < lower_keltner[i] and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below middle of Keltner Channel or squeeze fires in opposite direction
            if close[i] < ma[i] or (squeeze[i] and close[i] < lower_keltner[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above middle of Keltner Channel or squeeze fires in opposite direction
            if close[i] > ma[i] or (squeeze[i] and close[i] > upper_keltner[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
# 4h_Keltner_Channel_Squeeze_Breakout_1dTrend_Volume
# Hypothesis: Breakouts from low-volatility periods (Keltner channel squeeze) in the direction of 1d EMA trend, confirmed by volume.
# Works in bull markets by catching breakouts upward; in bear markets by catching breakdowns downward.
# The squeeze filter reduces whipsaws, and volume confirmation ensures conviction.
# Target: 20-40 trades/year to stay within optimal trade frequency for 4h.

name = "4h_Keltner_Channel_Squeeze_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Keltner Channel parameters
    kc_period = 20
    atr_period = 10
    kc_multiplier = 1.5
    
    # True Range and ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Keltner Channel
    ma = pd.Series(close).ewm(span=kc_period, adjust=False, min_periods=kc_period).mean().values
    upper_keltner = ma + kc_multiplier * atr
    lower_keltner = ma - kc_multiplier * atr
    
    # Bollinger Bands for squeeze detection
    bb_period = 20
    bb_std = 2.0
    bb_ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = bb_ma + bb_std * bb_std_dev
    lower_bb = bb_ma - bb_std * bb_std_dev
    
    # Squeeze condition: Bollinger Bands inside Keltner Channel
    squeeze = (upper_bb <= upper_keltner) & (lower_bb >= lower_keltner)
    
    # 1d EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ma[i]) or np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: squeeze breakout up, price above upper Keltner, 1d EMA uptrend, volume confirmation
            if squeeze[i-1] and close[i] > upper_keltner[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: squeeze breakout down, price below lower Keltner, 1d EMA downtrend, volume confirmation
            elif squeeze[i-1] and close[i] < lower_keltner[i] and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below middle of Keltner Channel or squeeze fires in opposite direction
            if close[i] < ma[i] or (squeeze[i] and close[i] < lower_keltner[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above middle of Keltner Channel or squeeze fires in opposite direction
            if close[i] > ma[i] or (squeeze[i] and close[i] > upper_keltner[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
# 4h_Keltner_Channel_Squeeze_Breakout_1dTrend_Volume
# Hypothesis: Breakouts from low-volatility periods (Keltner channel squeeze) in the direction of 1d EMA trend, confirmed by volume.
# Works in bull markets by catching breakouts upward; in bear markets by catching breakdowns downward.
# The squeeze filter reduces whipsaws, and volume confirmation ensures conviction.
# Target: 20-40 trades/year to stay within optimal trade frequency for 4h.

name = "4h_Keltner_Channel_Squeeze_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Keltner Channel parameters
    kc_period = 20
    atr_period = 10
    kc_multiplier = 1.5
    
    # True Range and ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Keltner Channel
    ma = pd.Series(close).ewm(span=kc_period, adjust=False, min_periods=kc_period).mean().values
    upper_keltner = ma + kc_multiplier * atr
    lower_keltner = ma - kc_multiplier * atr
    
    # Bollinger Bands for squeeze detection
    bb_period = 20
    bb_std = 2.0
    bb_ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = bb_ma + bb_std * bb_std_dev
    lower_bb = bb_ma - bb_std * bb_std_dev
    
    # Squeeze condition: Bollinger Bands inside Keltner Channel
    squeeze = (upper_bb <= upper_keltner) & (lower_bb >= lower_keltner)
    
    # 1d EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ma[i]) or np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: squeeze breakout up, price above upper Keltner, 1d EMA uptrend, volume confirmation
            if squeeze[i-1] and close[i] > upper_keltner[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: squeeze breakout down, price below lower Keltner, 1d EMA downtrend, volume confirmation
            elif squeeze[i-1] and close[i] < lower_keltner[i] and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below middle of Keltner Channel or squeeze fires in opposite direction
            if close[i] < ma[i] or (squeeze[i] and close[i] < lower_keltner[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above middle of Keltner Channel or squeeze fires in opposite direction
            if close[i] > ma[i] or (squeeze[i] and close[i] > upper_keltner[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
# 4h_Keltner_Channel_Squeeze_Breakout_1dTrend_Volume
# Hypothesis: Breakouts from low-volatility periods (Keltner channel squeeze) in the direction of 1d EMA trend, confirmed by volume.
# Works in bull markets by catching breakouts upward; in bear markets by catching breakdowns downward.
# The squeeze filter reduces whipsaws, and volume confirmation ensures conviction.
# Target: 20-40 trades/year to stay within optimal trade frequency for 4h.

name = "4h_Keltner_Channel_Squeeze_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Keltner Channel parameters
    kc_period = 20
    atr_period = 10
    kc_multiplier = 1.5
    
    # True Range and ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Keltner Channel
    ma = pd.Series(close).ewm(span=kc_period, adjust=False, min_periods=kc_period).mean().values
    upper_keltner = ma + kc_multiplier * atr
    lower_keltner = ma - kc_multiplier * atr
    
    # Bollinger Bands for squeeze detection
    bb_period = 20
    bb_std = 2.0
    bb_ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = bb_ma + bb_std * bb_std_dev
    lower_bb = bb_ma - bb_std * bb_std_dev
    
    # Squeeze condition: Bollinger Bands inside Keltner Channel
    squeeze = (upper_bb <= upper_keltner) & (lower_bb >= lower_keltner)
    
    # 1d EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ma[i]) or np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: squeeze breakout up, price above upper Keltner, 1d EMA uptrend, volume confirmation
            if squeeze[i-1] and close[i] > upper_keltner[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: squeeze breakout down, price below lower Keltner, 1d EMA downtrend, volume confirmation
            elif squeeze[i-1] and close[i] < lower_keltner[i] and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below middle of Keltner Channel or squeeze fires in opposite direction
            if close[i] < ma[i] or (squeeze[i] and close[i] < lower_keltner[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above middle of Keltner Channel or squeeze fires in opposite direction
            if close[i] > ma[i] or (squeeze[i] and close[i] > upper_keltner[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
# 4h_Keltner_Channel_Squeeze_Breakout_1dTrend_Volume
# Hypothesis: Breakouts from low-volatility periods (Keltner channel squeeze) in the direction of 1d EMA trend, confirmed by volume.
# Works in bull markets by catching breakouts upward; in bear markets by catching breakdowns downward.
# The squeeze filter reduces whipsaws, and volume confirmation ensures conviction.
# Target: 20-40 trades/year to stay within optimal trade frequency for 4h.

name = "4h_Keltner_Channel_Squeeze_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Keltner Channel parameters
    kc_period = 20
    atr_period = 10
    kc_multiplier = 1.5
    
    # True Range and ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Keltner Channel
    ma = pd.Series(close).ewm(span=kc_period, adjust=False, min_periods=kc_period).mean().values
    upper_keltner = ma + kc_multiplier * atr
    lower_keltner = ma - kc_multiplier * atr
    
    # Bollinger Bands for squeeze detection
    bb_period = 20
    bb_std = 2.0
    bb_ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = bb_ma + bb_std * bb_std_dev
    lower_bb = bb_ma - bb_std * bb_std_dev
    
    # Squeeze condition: Bollinger Bands inside Keltner Channel
    squeeze = (upper_bb <= upper_keltner) & (lower_bb >= lower_keltner)
    
    # 1d EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ma[i]) or np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: squeeze breakout up, price above upper Keltner, 1d EMA uptrend, volume confirmation
            if squeeze[i-1] and close[i] > upper_keltner[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: squeeze breakout down, price below lower Keltner, 1d EMA downtrend, volume confirmation
            elif squeeze[i-1] and close[i] < lower_keltner[i] and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below middle of Keltner Channel or squeeze fires in opposite direction
            if close[i] < ma[i] or (squeeze[i] and close[i] < lower_keltner[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above middle of Keltner Channel or squeeze fires in opposite direction
            if close[i] > ma[i] or (squeeze[i] and close[i] > upper_keltner[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
# 4h_Keltner_Channel_Squeeze_Breakout_1dTrend_Volume
# Hypothesis: Breakouts from low-volatility periods (Keltner channel squeeze) in the direction of 1d EMA trend, confirmed by volume.
# Works in bull markets by catching breakouts upward; in bear markets by catching breakdowns downward.
# The squeeze filter reduces whipsaws, and volume confirmation ensures conviction.
# Target: 20-40 trades/year to stay within optimal trade frequency for 4h.

name = "4h_Keltner_Channel_Squeeze_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Keltner Channel parameters
    kc_period = 20
    atr_period = 10
    kc_multiplier = 1.5
    
    # True Range and ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Keltner Channel
    ma = pd.Series(close).ewm(span=kc_period, adjust=False, min_periods=kc_period).mean().values
    upper_keltner = ma + kc_multiplier * atr
    lower_keltner = ma - kc_multiplier * atr
    
    # Bollinger Bands for squeeze detection
    bb_period = 20
    bb_std = 2.0
    bb_ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = bb_ma + bb_std * bb_std_dev
    lower_bb = bb_ma - bb_std * bb_std_dev
    
    # Squeeze condition: Bollinger Bands inside Keltner Channel
    squeeze = (upper_bb <= upper_keltner) & (lower_bb >= lower_keltner)
    
    # 1d EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ma[i]) or np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: squeeze breakout up, price above upper Keltner, 1d EMA uptrend, volume confirmation
            if squeeze[i-1] and close[i] > upper_keltner[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: squeeze