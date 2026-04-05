#!/usr/bin/env python3
"""
exp_7574_1h_4h_1d_momentum_v1
Hypothesis: 1-hour momentum strategy with 4-hour and 1-day trend filters.
Uses 4-hour RSI(14) for trend direction (above 50 = bullish, below 50 = bearish) and
1-day RSI(14) for long-term trend confirmation. Enters on 1-hour RSI(14) pullbacks
in the direction of the higher timeframe trends during active session (08-20 UTC).
Uses volume confirmation (1.5x average volume) to filter false signals.
Target: 60-150 total trades over 4 years (15-37/year) with strict entry conditions.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7574_1h_4h_1d_momentum_v1"
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
SESSION_START_HOUR = 8
SESSION_END_HOUR = 20

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h RSI for trend direction
    close_4h = df_4h['close'].values
    delta_4h = pd.Series(close_4h).diff()
    gain_4h = delta_4h.where(delta_4h > 0, 0)
    loss_4h = -delta_4h.where(delta_4h < 0, 0)
    avg_gain_4h = pd.Series(gain_4h).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    avg_loss_4h = pd.Series(loss_4h).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    rs_4h = avg_gain_4h / avg_loss_4h.replace(0, np.inf)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h = rsi_4h.values
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Calculate 1d RSI for long-term trend
    close_1d = df_1d['close'].values
    delta_1d = pd.Series(close_1d).diff()
    gain_1d = delta_1d.where(delta_1d > 0, 0)
    loss_1d = -delta_1d.where(delta_1d < 0, 0)
    avg_gain_1d = pd.Series(gain_1d).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    avg_loss_1d = pd.Series(loss_1d).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    rs_1d = avg_gain_1d / avg_loss_1d.replace(0, np.inf)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h RSI for entry timing
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for stop management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(alpha=1/ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Session filter
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD * 2, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(rsi_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Check session
        hour = hours[i]
        in_session = SESSION_START_HOUR <= hour <= SESSION_END_HOUR
        
        # Check stoploss
        if position == 1 and close[i] <= entry_price - (ATR_STOP_MULTIPLIER * atr[i]):
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] >= entry_price + (ATR_STOP_MULTIPLIER * atr[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Determine trends
        bullish_4h = rsi_4h_aligned[i] > 50
        bearish_4h = rsi_4h_aligned[i] < 50
        bullish_1d = rsi_1d_aligned[i] > 50
        bearish_1d = rsi_1d_aligned[i] < 50
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions: RSI pullback in direction of trend
        long_entry = (
            bullish_4h and bullish_1d and  # Both timeframes bullish
            rsi[i] < RSI_OVERSOLD and      # Oversold on 1h
            volume_confirmed and
            in_session
        )
        short_entry = (
            bearish_4h and bearish_1d and  # Both timeframes bearish
            rsi[i] > RSI_OVERBOUGHT and    # Overbought on 1h
            volume_confirmed and
            in_session
        )
        
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
exp_7574_1h_4h_1d_momentum_v1
Hypothesis: 1-hour momentum strategy with 4-hour and 1-day trend filters.
Uses 4-hour RSI(14) for trend direction (above 50 = bullish, below 50 = bearish) and
1-day RSI(14) for long-term trend confirmation. Enters on 1-hour RSI(14) pullbacks
in the direction of the higher timeframe trends during active session (08-20 UTC).
Uses volume confirmation (1.5x average volume) to filter false signals.
Target: 60-150 total trades over 4 years (15-37/year) with strict entry conditions.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7574_1h_4h_1d_momentum_v1"
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
SESSION_START_HOUR = 8
SESSION_END_HOUR = 20

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h RSI for trend direction
    close_4h = df_4h['close'].values
    delta_4h = pd.Series(close_4h).diff()
    gain_4h = delta_4h.where(delta_4h > 0, 0)
    loss_4h = -delta_4h.where(delta_4h < 0, 0)
    avg_gain_4h = pd.Series(gain_4h).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    avg_loss_4h = pd.Series(loss_4h).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    rs_4h = avg_gain_4h / avg_loss_4h.replace(0, np.inf)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h = rsi_4h.values
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Calculate 1d RSI for long-term trend
    close_1d = df_1d['close'].values
    delta_1d = pd.Series(close_1d).diff()
    gain_1d = delta_1d.where(delta_1d > 0, 0)
    loss_1d = -delta_1d.where(delta_1d < 0, 0)
    avg_gain_1d = pd.Series(gain_1d).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    avg_loss_1d = pd.Series(loss_1d).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    rs_1d = avg_gain_1d / avg_loss_1d.replace(0, np.inf)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h RSI for entry timing
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for stop management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(alpha=1/ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Session filter
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD * 2, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(rsi_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Check session
        hour = hours[i]
        in_session = SESSION_START_HOUR <= hour <= SESSION_END_HOUR
        
        # Check stoploss
        if position == 1 and close[i] <= entry_price - (ATR_STOP_MULTIPLIER * atr[i]):
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] >= entry_price + (ATR_STOP_MULTIPLIER * atr[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Determine trends
        bullish_4h = rsi_4h_aligned[i] > 50
        bearish_4h = rsi_4h_aligned[i] < 50
        bullish_1d = rsi_1d_aligned[i] > 50
        bearish_1d = rsi_1d_aligned[i] < 50
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions: RSI pullback in direction of trend
        long_entry = (
            bullish_4h and bullish_1d and  # Both timeframes bullish
            rsi[i] < RSI_OVERSOLD and      # Oversold on 1h
            volume_confirmed and
            in_session
        )
        short_entry = (
            bearish_4h and bearish_1d and  # Both timeframes bearish
            rsi[i] > RSI_OVERBOUGHT and    # Overbought on 1h
            volume_confirmed and
            in_session
        )
        
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