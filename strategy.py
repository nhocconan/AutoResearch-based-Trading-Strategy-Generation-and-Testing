# 1d 1w donchian breakout with volume confirmation and ATR filter
# Uses weekly trend direction from Donchian(20) on 1w timeframe
# Enters only on daily Donchian breakout in direction of weekly trend
# Filters with volume > 1.5x 20-period average and low volatility (ATR rank < 30)
# Exits on opposite Donchian breakout
# Position size 0.25 to limit drawdown
# Target: 15-30 trades/year per symbol to minimize fee drag

name = "1d_1w_donchian_trend_v3"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Load daily data ONCE before loop (for ATR)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channel (20-period) for trend
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    donch_high_1w = np.full(len(df_1w), np.nan)
    donch_low_1w = np.full(len(df_1w), np.nan)
    
    for i in range(20, len(df_1w)):
        donch_high_1w[i] = np.max(high_1w[i-20:i])
        donch_low_1w[i] = np.min(low_1w[i-20:i])
    
    # Align weekly Donchian to daily timeframe (only use completed weekly bars)
    donch_high_1d = align_htf_to_ltf(prices, df_1w, donch_high_1w)
    donch_low_1d = align_htf_to_ltf(prices, df_1w, donch_low_1w)
    
    # Calculate weekly trend: 1 if price above mid, -1 if below, 0 if inside
    weekly_trend = np.zeros(len(df_1w))
    for i in range(len(df_1w)):
        if not np.isnan(donch_high_1w[i]) and not np.isnan(donch_low_1w[i]):
            mid = (donch_high_1w[i] + donch_low_1w[i]) / 2
            if close_1w := df_1w['close'].values[i] > mid:
                weekly_trend[i] = 1
            elif close_1w < mid:
                weekly_trend[i] = -1
            else:
                weekly_trend[i] = 0
    
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # Calculate daily ATR (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr_1d = np.zeros(len(df_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr0 = high_1d[i] - low_1d[i]
        tr1 = abs(high_1d[i] - close_1d[i-1])
        tr2 = abs(low_1d[i] - close_1d[i-1])
        tr_1d[i] = max(tr0, tr1, tr2)
    
    atr_1d = np.zeros(len(df_1d))
    atr_1d[0] = tr_1d[0]
    for i in range(1, len(df_1d)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # ATR percentile rank (50-day lookback)
    atr_rank_1d = np.zeros(len(df_1d))
    for i in range(50, len(df_1d)):
        window = atr_1d[i-50:i]
        atr_rank_1d[i] = np.sum(window < atr_1d[i]) / len(window) * 100
    
    # Align ATR rank to daily timeframe
    atr_rank_aligned = align_htf_to_ltf(prices, df_1d, atr_rank_1d)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    # Calculate daily Donchian channel (20-period)
    donch_high_d = np.full(n, np.nan)
    donch_low_d = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high_d[i] = np.max(high[i-20:i])
        donch_low_d[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after ATR rank warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high_1d[i]) or 
            np.isnan(donch_low_1d[i]) or 
            np.isnan(weekly_trend_aligned[i]) or 
            np.isnan(atr_rank_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or
            np.isnan(donch_high_d[i]) or
            np.isnan(donch_low_d[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in low volatility environment (ATR rank < 30 = bottom 30% volatility)
        if atr_rank_aligned[i] >= 30:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below daily Donchian low
            if close[i] <= donch_low_d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above daily Donchian high
            if close[i] >= donch_high_d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above daily Donchian high with volume confirmation
            # AND weekly trend is up (1)
            vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
            if (close[i] > donch_high_d[i] and 
                vol_ratio > 1.5 and
                weekly_trend_aligned[i] == 1):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below daily Donchian low with volume confirmation
            # AND weekly trend is down (-1)
            elif (close[i] < donch_low_d[i] and 
                  vol_ratio > 1.5 and
                  weekly_trend_aligned[i] == -1):
                position = -1
                signals[i] = -0.25
    
    return signals