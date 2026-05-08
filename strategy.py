# 1d_ElderRay_1wTrend_Volume
# Hypothesis: Elder Ray index (Bull Power/Bear Power) on 1d with 1w trend filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long when Bull Power > 0, Bear Power < 0, 1w EMA20 > 1w EMA50 (uptrend), and volume > 1.5x 20-day average.
# Short when Bear Power < 0, Bull Power < 0, 1w EMA20 < 1w EMA50 (downtrend), and volume > 1.5x 20-day average.
# Exit when Bull Power and Bear Power cross zero or trend changes.
# Uses Elder Ray to measure bull/bear strength with trend filter to avoid false signals.
# Target: 30-80 total trades over 4 years (7-20/year) for low fee drift.

name = "1d_ElderRay_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Elder Ray on 1d: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1w data for trend filter: EMA20 and EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMAs to 1d timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0, Bear Power < 0, 1w uptrend, volume spike
            long_cond = (bull_power[i] > 0) and (bear_power[i] < 0) and \
                        (ema20_1w_aligned[i] > ema50_1w_aligned[i]) and volume_filter[i]
            # Short conditions: Bear Power < 0, Bull Power < 0, 1w downtrend, volume spike
            short_cond = (bear_power[i] < 0) and (bull_power[i] < 0) and \
                         (ema20_1w_aligned[i] < ema50_1w_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 or Bear Power >= 0 or trend turns down
            if (bull_power[i] <= 0) or (bear_power[i] >= 0) or \
               (ema20_1w_aligned[i] <= ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power >= 0 or Bull Power > 0 or trend turns up
            if (bear_power[i] >= 0) or (bull_power[i] > 0) or \
               (ema20_1w_aligned[i] >= ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals