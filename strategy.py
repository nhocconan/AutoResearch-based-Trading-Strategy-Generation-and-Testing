# 1d_Camarilla_R1S1_Breakout_WeeklyTrend
# Hypothesis: Daily strategy using Camarilla pivot levels from weekly timeframe with daily volume confirmation.
# Weekly R1/S1 provide institutional support/resistance. Long when price breaks above weekly R1 with volume confirmation and weekly trend alignment.
# Short when price breaks below weekly S1 with volume confirmation and weekly trend alignment.
# Uses daily volume > 1.5x 20-period EMA for confirmation.
# Weekly trend filter using 50-period EMA on weekly close.
# Designed for low trade frequency (10-20/year) to minimize fee drag while capturing institutional level breaks.

name = "1d_Camarilla_R1S1_Breakout_WeeklyTrend"
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
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels from previous week's range (R1/S1 only)
    camarilla_r1 = np.zeros_like(close_1w)
    camarilla_s1 = np.zeros_like(close_1w)
    
    for i in range(1, len(close_1w)):
        # Previous week's high, low, close
        ph = high_1w[i-1]
        pl = low_1w[i-1]
        pc = close_1w[i-1]
        
        # Range
        rng = ph - pl
        
        # Camarilla R1 and S1 levels
        camarilla_r1[i] = pc + (rng * 1.1 / 6)
        camarilla_s1[i] = pc - (rng * 1.1 / 6)
    
    # First week has no previous data
    camarilla_r1[0] = camarilla_s1[0] = np.nan
    
    # Align Camarilla levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Daily volume confirmation: volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    # Weekly trend filter: 50-period EMA on weekly close
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA(50)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Breakout entries: price breaks R1 or S1 with volume confirmation and trend alignment
            if close[i] > r1_aligned[i] and vol_confirm[i]:
                # Only take long breakout if above weekly EMA50 (uptrend)
                if close[i] > ema_50_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            elif close[i] < s1_aligned[i] and vol_confirm[i]:
                # Only take short breakout if below weekly EMA50 (downtrend)
                if close[i] < ema_50_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: break below S1 or trend turns down
            if close[i] < s1_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above R1 or trend turns up
            if close[i] > r1_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals