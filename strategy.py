# 12h_CamarillaBreakout_1wTrend_Volume
# Hypothesis: 12h Camarilla Pivot Breakout with 1 week Trend and Volume Spike
# - Uses Camarilla levels from weekly timeframe (S1/S2 for long, R1/R2 for short)
# - Breakout above S1 with 1w uptrend or below R1 with 1w downtrend
# - Volume spike confirms breakout strength
# - Weekly trend filter ensures we trade with the dominant weekly trend
# - Target: 15-30 trades/year to minimize fee drag on 12h timeframe

name = "12h_CamarillaBreakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels using previous week's data
    # S1 = C - (H-L)*1.08, S2 = C - (H-L)*1.16, R1 = C + (H-L)*1.08, R2 = C + (H-L)*1.16
    n1w = len(close_1w)
    camarilla_S1 = np.full(n1w, np.nan)
    camarilla_S2 = np.full(n1w, np.nan)
    camarilla_R1 = np.full(n1w, np.nan)
    camarilla_R2 = np.full(n1w, np.nan)
    
    for i in range(1, n1w):
        H = high_1w[i-1]
        L = low_1w[i-1]
        C = close_1w[i-1]
        range_val = H - L
        camarilla_S1[i] = C - range_val * 1.08
        camarilla_S2[i] = C - range_val * 1.16
        camarilla_R1[i] = C + range_val * 1.08
        camarilla_R2[i] = C + range_val * 1.16
    
    # Align Camarilla levels to 12h timeframe
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_S1)
    camarilla_S2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_S2)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_R1)
    camarilla_R2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_R2)
    
    # 1w data for trend filter (EMA34)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_S1_aligned[i]) or np.isnan(camarilla_S2_aligned[i]) or 
            np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_R2_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above S1 (support) with 1w uptrend + volume spike
            long_cond = (close[i] > camarilla_S1_aligned[i] and 
                        ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below R1 (resistance) with 1w downtrend + volume spike
            short_cond = (close[i] < camarilla_R1_aligned[i] and 
                         ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S2 (strong support break)
            if close[i] < camarilla_S2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R2 (strong resistance break)
            if close[i] > camarilla_R2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals