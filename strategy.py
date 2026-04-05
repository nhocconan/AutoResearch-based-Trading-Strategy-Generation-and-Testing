#!/usr/bin/env python3
"""
Experiment #8194: 1-hour RSI mean reversion with 4h trend filter and volume confirmation.
Hypothesis: In both bull and bear markets, price tends to revert to the mean during pullbacks within the prevailing trend.
We use 4h RSI for trend direction (above 50 = bullish, below 50 = bearish) and 1h RSI for mean reversion entries.
Only take long when 4h trend is bullish and 1h RSI < 30 (oversold), short when 4h trend bearish and 1h RSI > 70 (overbought).
Volume confirmation requires current volume > 1.5x 20-period MA to ensure institutional participation.
This reduces false signals in sideways markets and captures reversion moves within trends.
Target: 60-150 total trades over 4 years (15-37/year) to balance signal quality and fee drag.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8194_1h_rsi_meanrev_4htrend_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h RSI for trend direction
    close_4h = df_4h['close'].values
    delta_4h = np.diff(close_4h, prepend=close_4h[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    avg_gain_4h = pd.Series(gain_4h).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    avg_loss_4h = pd.Series(loss_4h).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    rs_4h = avg_gain_4h / (avg_loss_4h + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h_trend = np.where(rsi_4h > 50, 1, -1)  # 1=bullish trend, -1=bearish trend
    rsi_4h_trend_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h_trend)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h RSI for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(rsi_4h_trend_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - (ATR_STOP_MULTIPLIER * atr[i]):
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + (ATR_STOP_MULTIPLIER * atr[i]):
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from 4h RSI trend
        bull_bias = rsi_4h_trend_aligned[i] == 1   # 4h RSI > 50 = bullish trend
        bear_bias = rsi_4h_trend_aligned[i] == -1  # 4h RSI < 50 = bearish trend
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Mean reversion conditions
        long_entry = bull_bias and (rsi[i] < RSI_OVERSOLD) and volume_confirmed
        short_entry = bear_bias and (rsi[i] > RSI_OVERBOUGHT) and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #8194: 1-hour Bollinger Band mean reversion with 4h trend filter and volume confirmation.
Hypothesis: Price tends to revert to the mean when touching Bollinger Bands during pullbacks within the prevailing trend.
We use 4h ADX for trend strength (ADX > 25 = trending) and 4h EMA50 for trend direction.
Only take long when 4h trend is bullish and price touches lower BB, short when bearish and price touches upper BB.
Volume confirmation requires current volume > 1.5x 20-period MA to ensure institutional participation.
This reduces false signals in sideways markets and captures reversion moves within trends.
Target: 60-150 total trades over 4 years (15-37/year) to balance signal quality and fee drag.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8194_1h_bb_meanrev_4hadx_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
BB_PERIOD = 20
BB_STD = 2.0
ADX_PERIOD = 14
EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h ADX for trend strength
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(np.roll(high_4h, 1) - close_4h)
    tr3_4h = np.abs(np.roll(low_4h, 1) - close_4h)
    tr_4h = np.maximum(np.maximum(tr1_4h, tr2_4h), tr3_4h)
    
    # Directional Movement
    dm_plus_4h = np.where((high_4h - np.roll(high_4h, 1)) > (np.roll(low_4h, 1) - low_4h), 
                          np.maximum(high_4h - np.roll(high_4h, 1), 0), 0)
    dm_minus_4h = np.where((np.roll(low_4h, 1) - low_4h) > (high_4h - np.roll(high_4h, 1)), 
                           np.maximum(np.roll(low_4h, 1) - low_4h, 0), 0)
    
    # Smooth TR and DM
    tr_ma_4h = pd.Series(tr_4h).ewm(alpha=1/ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    dm_plus_ma_4h = pd.Series(dm_plus_4h).ewm(alpha=1/ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    dm_minus_ma_4h = pd.Series(dm_minus_4h).ewm(alpha=1/ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # Directional Indicators
    di_plus_4h = 100 * dm_plus_ma_4h / (tr_ma_4h + 1e-10)
    di_minus_4h = 100 * dm_minus_ma_4h / (tr_ma_4h + 1e-10)
    
    # DX and ADX
    dx_4h = 100 * np.abs(di_plus_4h - di_minus_4h) / (di_plus_4h + di_minus_4h + 1e-10)
    adx_4h = pd.Series(dx_4h).ewm(alpha=1/ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # 4h EMA50 for trend direction
    ema_4h = pd.Series(close_4h).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    trend_dir_4h = np.where(close_4h > ema_4h, 1, -1)  # 1=bullish, -1=bearish
    
    # Combine trend strength and direction: only trade when ADX > 25
    trend_signal_4h = np.where(adx_4h > 25, trend_dir_4h, 0)
    trend_signal_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_signal_4h)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands
    sma = pd.Series(close).rolling(window=BB_PERIOD, min_periods=BB_PERIOD).mean().values
    std = pd.Series(close).rolling(window=BB_PERIOD, min_periods=BB_PERIOD).std().values
    upper_band = sma + (BB_STD * std)
    lower_band = sma - (BB_STD * std)
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(BB_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, ADX_PERIOD, EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(trend_signal_4h_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - (ATR_STOP_MULTIPLIER * atr[i]):
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + (ATR_STOP_MULTIPLIER * atr[i]):
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from 4h trend
        bull_bias = trend_signal_4h_aligned[i] == 1   # 4h trend bullish and strong
        bear_bias = trend_signal_4h_aligned[i] == -1  # 4h trend bearish and strong
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Bollinger Band mean reversion conditions
        long_entry = bull_bias and (close[i] <= lower_band[i]) and volume_confirmed
        short_entry = bear_bias and (close[i] >= upper_band[i]) and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #8194: 1-hour Donchian breakout with 4h trend filter and volume confirmation.
Hypothesis: Price breaking beyond 20-period high/low on 1h with volume >1.5x 20-period MA 
and aligned 4h trend (price above/below 4h EMA50) captures sustained moves while avoiding 
whipsaw in both bull and bear markets. The 4h trend filter provides stronger trend context 
than 1h alone, reducing false breakouts during consolidation periods. 
Targeting 60-150 total trades over 4 years for optimal balance of signal quality and cost.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8194_1h_donchian20_4h_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
EMA_PERIOD = 50
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # Price relative to EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_4h > ema_4h, 1, -1)  # 1=bullish, -1=bearish
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_4h, price_vs_ema)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Price channel (Donchian)
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from 4h EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 4h close above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 4h close below EMA50
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions - require close beyond channel bands to avoid wicks
        upper_breakout = (close[i] > highest_high[i-1]) and (i-1 >= 0) and not np.isnan(highest_high[i-1])
        lower_breakout = (close[i] < lowest_low[i-1]) and (i-1 >= 0) and not np.isnan(lowest_low[i-1])
        
        # Entry conditions
        long_entry = bull_bias and upper_breakout and volume_confirmed
        short_entry = bear_bias and lower_breakout and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (2.0 * atr[i])  # 2:1 reward/risk
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (2.0 * atr[i])  # 2:1 reward/risk
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #8194: 1-hour RSI mean reversion with 4h trend filter and volume confirmation.
Hypothesis: In both bull and bear markets, price tends to revert to the mean during pullbacks within the prevailing trend.
We use 4h RSI for trend direction (above 50 = bullish, below 50 = bearish) and 1h RSI for mean reversion entries.
Only take long when 4h trend is bullish and 1h RSI < 30 (oversold), short when 4h trend bearish and 1h RSI > 70 (overbought).
Volume confirmation requires current volume > 1.5x 20-period MA to ensure institutional participation.
This reduces false signals in sideways markets and captures reversion moves within trends.
Target: 60-150 total trades over 4 years (15-37/year) to balance signal quality and fee drag.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8194_1h_rsi_meanrev_4htrend_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h RSI for trend direction
    close_4h = df_4h['close'].values
    delta_4h = np.diff(close_4h, prepend=close_4h[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    avg_gain_4h = pd.Series(gain_4h).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    avg_loss_4h = pd.Series(loss_4h).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    rs_4h = avg_gain_4h / (avg_loss_4h + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h_trend = np.where(rsi_4h > 50, 1, -1)  # 1=bullish trend, -1=bearish trend
    rsi_4h_trend_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h_trend)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h RSI for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(rsi_4h_trend_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - (ATR_STOP_MULTIPLIER * atr[i]):
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + (ATR_STOP_MULTIPLIER * atr[i]):
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from 4h RSI trend
        bull_bias = rsi_4h_trend_aligned[i] == 1   # 4h RSI > 50 = bullish trend
        bear_bias = rsi_4h_trend_aligned[i] == -1  # 4h RSI < 50 = bearish trend
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Mean reversion conditions
        long_entry = bull_bias and (rsi[i] < RSI_OVERSOLD) and volume_confirmed
        short_entry = bear_bias and (rsi[i] > RSI_OVERBOUGHT) and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

--- END OF STRATEGY ---