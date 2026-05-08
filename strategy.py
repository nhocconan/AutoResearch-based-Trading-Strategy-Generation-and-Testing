# 1d_GoldenCross_BitcoinCycle
# Hypothesis: 1d golden cross (SMA50 crossing SMA200) with volume confirmation and 1w trend filter.
# Works in bull markets via trend following; works in bear markets via avoiding false signals during downtrends.
# Uses only two moving averages for clear, infrequent signals to minimize fee drag.
# Target: 20-50 total trades over 4 years (5-12/year) to stay under fee drag threshold.

name = "1d_GoldenCross_BitcoinCycle"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate SMA50 and SMA200
    sma50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    sma200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate SMA50 on 1w close for trend filter
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Need enough data for SMA200
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(sma50[i]) or np.isnan(sma200[i]) or 
            np.isnan(sma50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        sma50_val = sma50[i]
        sma200_val = sma200[i]
        sma50_1w_val = sma50_1w_aligned[i]
        vol_conf_val = vol_conf[i]
        
        # Golden cross: SMA50 crosses above SMA200
        golden_cross = sma50_val > sma200_val and (i == start_idx or sma50[i-1] <= sma200[i-1])
        # Death cross: SMA50 crosses below SMA200
        death_cross = sma50_val < sma200_val and (i == start_idx or sma50[i-1] >= sma200[i-1])
        
        if position == 0:
            # Enter long: golden cross, price above SMA200, 1w uptrend, volume confirmation
            if golden_cross and close_val > sma200_val and sma50_1w_val > close_1w[-1] if len(close_1w) > 0 else False and vol_conf_val:
                signals[i] = 0.25
                position = 1
        elif position == 1:
            # Exit long: death cross or price below SMA200 or 1w trend down
            if death_cross or close_val < sma200_val or sma50_1w_val < close_1w[-1] if len(close_1w) > 0 else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals