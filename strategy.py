#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and volume confirmation
# Enter long when: RSI(14) > 55, price > 4h EMA(20), volume > 1.3x avg, during active session (08-20 UTC)
# Enter short when: RSI(14) < 45, price < 4h EMA(20), volume > 1.3x avg, during active session
# Exit when RSI crosses opposite threshold (long exit at RSI<45, short exit at RSI>55)
# Uses 4h trend to filter counter-trend trades, targeting 80-120 trades over 4 years

name = "1h_momentum_4h_ema_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14) on 1h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 4h EMA(20) for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_20 = pd.Series(close_4h).ewm(span=20, adjust=False).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_4h, ema_20)
    
    # Volume confirmation: volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.3 * volume_ma
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for EMA to stabilize
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(ema_20_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 1:  # long position
            # Exit: RSI < 45 (back to neutral)
            if rsi[i] < 45:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI > 55 (back to neutral)
            if rsi[i] > 55:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: RSI momentum + trend filter + volume + session
            if in_session and volume[i] > volume_threshold[i]:
                if rsi[i] > 55 and close[i] > ema_20_aligned[i]:
                    # Bullish momentum with 4h uptrend
                    signals[i] = 0.20
                    position = 1
                elif rsi[i] < 45 and close[i] < ema_20_aligned[i]:
                    # Bearish momentum with 4h downtrend
                    signals[i] = -0.20
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h breakout with 4h/1d trend filter and volume spike
# Enter long when: price > 4h Donchian(20) high, volume > 2x 20-period avg, during active session (08-20 UTC)
# Enter short when: price < 4h Donchian(20) low, volume > 2x 20-period avg, during active session
# Exit when price crosses opposite Donchian band or RSI(14) reaches extreme (70/30)
# Uses multi-timeframe trend structure to filter false breakouts, targeting 60-120 trades over 4 years

name = "1h_breakout_4h_donchian_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian(20) for trend structure
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    # Calculate 20-period Donchian channels on 4h
    high_ma = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high = align_htf_to_ltf(prices, df_4h, high_ma)
    donchian_low = align_htf_to_ltf(prices, df_4h, low_ma)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    # RSI(14) for exit signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_threshold[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 1:  # long position
            # Exit: price < 4h Donchian low OR RSI > 70 (overbought)
            if close[i] < donchian_low[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price > 4h Donchian high OR RSI < 30 (oversold)
            if close[i] > donchian_high[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for breakout entries: price breaks 4h Donchian + volume spike + session
            if in_session and volume[i] > volume_threshold[i]:
                if close[i] > donchian_high[i]:
                    # Bullish breakout above 4h resistance
                    signals[i] = 0.20
                    position = 1
                elif close[i] < donchian_low[i]:
                    # Bearish breakdown below 4h support
                    signals[i] = -0.20
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h trend following with 1d ADX filter and volume confirmation
# Enter long when: ADX(14) > 25, +DI > -DI, price > 1d EMA(50), volume > 1.5x avg, during active session (08-20 UTC)
# Enter short when: ADX(14) > 25, -DI > +DI, price < 1d EMA(50), volume > 1.5x avg, during active session
# Exit when ADX < 20 (trend weakening) or opposite DI crossover
# Uses daily trend strength to filter whipsaws, targeting 50-100 trades over 4 years

name = "1h_trend_adx_1dema_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ADX(14) and DI on 1h
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_period = 14
    tr_sum = pd.Series(tr).rolling(window=tr_period, min_periods=tr_period).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=tr_period, min_periods=tr_period).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=tr_period, min_periods=tr_period).sum().values
    
    # Avoid division by zero
    tr_sum = np.where(tr_sum == 0, 1e-10, tr_sum)
    
    di_plus = 100 * dm_plus_sum / tr_sum
    di_minus = 100 * dm_minus_sum / tr_sum
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=tr_period, min_periods=tr_period).mean().values
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Wait for ADX to stabilize
        # Skip if required data not available
        if (np.isnan(adx[i]) or np.isnan(di_plus[i]) or np.isnan(di_minus[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 1:  # long position
            # Exit: ADX < 20 OR -DI > +DI
            if adx[i] < 20 or di_minus[i] > di_plus[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: ADX < 20 OR +DI > -DI
            if adx[i] < 20 or di_plus[i] > di_minus[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: strong trend + direction + price vs 1d EMA + volume + session
            if in_session and volume[i] > volume_threshold[i]:
                if adx[i] > 25 and di_plus[i] > di_minus[i] and close[i] > ema_50_aligned[i]:
                    # Strong uptrend with price above daily EMA
                    signals[i] = 0.20
                    position = 1
                elif adx[i] > 25 and di_minus[i] > di_plus[i] and close[i] < ema_50_aligned[i]:
                    # Strong downtrend with price below daily EMA
                    signals[i] = -0.20
                    position = -1
    
    return signals

---  --- & 0.240 \\
1h_breakout_4h_donchian_volume_session_v1 & 0.233 \\
1h_trend_adx_1dema_volume_session_v1 & 0.167 \\ \end{array} }  \end{aligned} }  \end{aligned}  \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \end{aligned} \