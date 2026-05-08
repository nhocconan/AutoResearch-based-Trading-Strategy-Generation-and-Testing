# 12h_RSI_4hSupertrend_Volume - 12h RSI with 4h Supertrend and volume confirmation
# Hypothesis: RSI mean-reversion on 12h (30/70) combined with 4h Supertrend direction filter and volume spike
# Works in both bull/bear: RSI captures reversals, Supertrend filters for trend alignment, volume confirms momentum
# Targets 15-30 trades per year (~60-120 total over 4 years) to minimize fee drag

name = "12h_RSI_4hSupertrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14) on 12h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 4h data for Supertrend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Supertrend(10, 3) on 4h
    atr_period = 10
    multiplier = 3
    
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = abs(df_4h['high'] - df_4h['close'].shift(1))
    tr3 = abs(df_4h['low'] - df_4h['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    
    hl2 = (df_4h['high'] + df_4h['low']) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize Supertrend
    supertrend = np.zeros(len(df_4h))
    direction = np.ones(len(df_4h))  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(df_4h)):
        if df_4h['close'].iloc[i] > upper_band[i-1]:
            direction[i] = 1
        elif df_4h['close'].iloc[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        supertrend[i] = lower_band[i] if direction[i] == 1 else upper_band[i]
    
    # Align Supertrend and direction to 12h
    supertrend_4h = supertrend
    direction_4h = direction
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend_4h)
    direction_aligned = align_htf_to_ltf(prices, df_4h, direction_4h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # RSI and Supertrend warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi[i]) or np.isnan(supertrend_aligned[i]) or 
            np.isnan(direction_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi[i]
        close_val = close[i]
        supertrend_val = supertrend_aligned[i]
        direction_val = direction_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: RSI < 30 (oversold), price above Supertrend (uptrend), volume confirmation
            if rsi_val < 30 and close_val > supertrend_val and direction_val == 1 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: RSI > 70 (overbought), price below Supertrend (downtrend), volume confirmation
            elif rsi_val > 70 and close_val < supertrend_val and direction_val == -1 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI > 50 (mean reversion) or price below Supertrend (trend change)
            if rsi_val > 50 or close_val < supertrend_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI < 50 (mean reversion) or price above Supertrend (trend change)
            if rsi_val < 50 or close_val > supertrend_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals