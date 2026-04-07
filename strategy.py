# 6h Bollinger Band Width Squeeze with RSI(2) Reversion and Volume Confirmation
# Long when BBW at 20-day low, RSI(2) < 10, and volume > 1.5x average
# Short when BBW at 20-day low, RSI(2) > 90, and volume > 1.5x average
# Exit when RSI(2) crosses 50 or BBW expands above 50-day average
# Uses 1d BBW for regime detection and 6s RSI(2) for entry timing
# Target: 75-200 total trades over 4 years (19-50/year)

name = "6s_bbw_squeeze_rsi2_vol_v1"
timeframe = "6s"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Bollinger Band Width regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate Bollinger Bands (20, 2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bbw = (upper_bb - lower_bb) / sma_20  # Bollinger Band Width
    bbw_20d_low = bbw.rolling(window=20, min_periods=20).min()  # 20-day lowest BBW
    bbw_50d_avg = bbw.rolling(window=50, min_periods=50).mean()  # 50-day average BBW
    
    # Align BBW indicators to 6s timeframe
    bbw_20d_low_aligned = align_htf_to_ltf(prices, df_1d, bbw_20d_low.values)
    bbw_50d_avg_aligned = align_htf_to_ltf(prices, df_1d, bbw_50d_avg.values)
    
    # 6s RSI(2) for entry signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Fill NaN with 50 (neutral)
    
    # 6s volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(bbw_20d_low_aligned[i]) or np.isnan(bbw_50d_avg_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Squeeze condition: BBW at 20-day low
        is_squeeze = bbw[i] <= bbw_20d_low_aligned[i] * 1.01  # Allow small tolerance
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI crosses above 50 or BBW expands significantly
            elif rsi[i] > 50 or bbw[i] > bbw_50d_avg_aligned[i] * 1.5:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI crosses below 50 or BBW expands significantly
            elif rsi[i] < 50 or bbw[i] > bbw_50d_avg_aligned[i] * 1.5:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in squeeze with RSI extremes and volume confirmation
            # Long: BBW squeeze, RSI(2) < 10 (oversold), volume spike
            if (is_squeeze and
                rsi[i] < 10 and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: BBW squeeze, RSI(2) > 90 (overbought), volume spike
            elif (is_squeeze and
                  rsi[i] > 90 and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals