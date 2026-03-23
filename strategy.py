#!/usr/bin/env python3
"""
Experiment #068: 30m Primary + 4h/1d HTF — Trend Pullback with Session + Volume + Funding

Hypothesis: 30m timeframe with 4h HMA trend bias + 30m RSI pullback entries, filtered by
UTC session (8-20), volume confirmation, and funding rate contrarian signal will generate
40-70 trades/year with Sharpe > 0.5. This avoids the overused CRSI+Chop combo.

Key innovations:
1) 4h HMA(21) for trend direction — proven in exp #061
2) 30m RSI(14) pullback: long when RSI 35-45 in uptrend, short when RSI 55-65 in downtrend
3) Session filter: only trade 8-20 UTC (reduces trades by ~60%)
4) Volume confirmation: volume > 0.8 * SMA(volume, 20)
5) Funding rate contrarian: z-score < -1.5 → favor longs, > +1.5 → favor shorts
6) 1d HMA for macro bias filter (avoid counter-macro trades)
7) ATR(14) stoploss at 2.5x with signal→0 exit

Why this should work:
- 30m TF with HTF trend = HTF trade frequency with 30m entry precision
- RSI pullback (not extreme) generates more reliable entries than CRSI extremes
- Session filter drastically reduces trade count (critical for lower TF)
- Funding rate adds edge specifically for BTC/ETH (SOL less affected)
- Discrete position sizes (0.25, 0.30) minimize fee churn

Position size: 0.25-0.30 (discrete)
Stoploss: 2.5*ATR trailing
Target: 40-70 trades/year, Sharpe > 0.5
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_pullback_4h1d_session_vol_funding_v1"
timeframe = "30m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_volume_ratio(volume, period=20):
    """Calculate volume relative to SMA."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = volume / (vol_sma + 1e-10)
    return ratio

def load_funding_data(symbol):
    """Load funding rate data for contrarian signal."""
    try:
        # Map symbol to funding file name
        symbol_map = {
            'BTCUSDT': 'BTCUSDT',
            'ETHUSDT': 'ETHUSDT',
            'SOLUSDT': 'SOLUSDT'
        }
        funding_symbol = symbol_map.get(symbol, symbol)
        funding_path = f"data/processed/funding/{funding_symbol}.parquet"
        df_funding = pd.read_parquet(funding_path)
        return df_funding
    except Exception:
        return None

def calculate_funding_zscore(funding_df, prices, window=30):
    """Calculate funding rate z-score aligned to prices."""
    if funding_df is None or len(funding_df) == 0:
        return np.zeros(len(prices))
    
    try:
        # Resample funding to 30m and align
        funding_df = funding_df.copy()
        funding_df['open_time'] = pd.to_datetime(funding_df['open_time'])
        funding_df = funding_df.set_index('open_time')
        
        # Get funding rate column
        if 'funding_rate' in funding_df.columns:
            fr_col = 'funding_rate'
        elif 'rate' in funding_df.columns:
            fr_col = 'rate'
        else:
            return np.zeros(len(prices))
        
        # Resample to 30m
        fr_30m = funding_df[fr_col].resample('30min').last()
        
        # Calculate z-score
        fr_roll_mean = fr_30m.rolling(window=window, min_periods=window).mean()
        fr_roll_std = fr_30m.rolling(window=window, min_periods=window).std()
        fr_zscore = (fr_30m - fr_roll_mean) / (fr_roll_std + 1e-10)
        fr_zscore = fr_zscore.fillna(0.0)
        
        # Align to prices
        prices_idx = prices.copy()
        prices_idx['open_time'] = pd.to_datetime(prices_idx['open_time'])
        prices_idx = prices_idx.set_index('open_time')
        
        # Reindex funding zscore to prices index
        fr_zscore_aligned = fr_zscore.reindex(prices_idx.index, method='ffill').fillna(0.0).values
        
        # Shift by 1 to avoid look-ahead
        fr_zscore_aligned = np.roll(fr_zscore_aligned, 1)
        fr_zscore_aligned[0] = 0.0
        
        return fr_zscore_aligned
    except Exception:
        return np.zeros(len(prices))

def get_utc_hour(prices, idx):
    """Get UTC hour for a given index."""
    try:
        open_time = prices['open_time'].iloc[idx]
        if isinstance(open_time, (int, np.integer)):
            # Unix timestamp in milliseconds
            dt = pd.Timestamp(open_time, unit='ms', tz='UTC')
        else:
            dt = pd.Timestamp(open_time, tz='UTC')
        return dt.hour
    except Exception:
        return 12  # Default to midday

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for intermediate trend
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d HMA for macro bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Load funding data (try, but don't fail if unavailable)
    try:
        symbol = prices.get('symbol', 'BTCUSDT')
        if isinstance(symbol, (list, np.ndarray)):
            symbol = symbol[0] if len(symbol) > 0 else 'BTCUSDT'
        funding_df = load_funding_data(symbol)
        funding_zscore = calculate_funding_zscore(funding_df, prices, window=30)
    except Exception:
        funding_zscore = np.zeros(n)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.30
    POSITION_SIZE_REDUCED = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_14[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(prices, i)
        in_session = 8 <= utc_hour <= 20
        
        # === HTF TREND BIAS ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirms = vol_ratio[i] > 0.8
        
        # === FUNDING RATE CONTRARIAN ===
        fund_z = funding_zscore[i] if i < len(funding_zscore) else 0.0
        funding_bullish = fund_z < -1.0  # Negative funding → long bias
        funding_bearish = fund_z > 1.0   # Positive funding → short bias
        
        # === RSI PULLBACK SIGNALS ===
        # Long: RSI 35-48 (pullback in uptrend, not oversold extreme)
        rsi_long_pullback = 35.0 <= rsi_14[i] <= 48.0
        # Short: RSI 52-65 (rally in downtrend, not overbought extreme)
        rsi_short_pullback = 52.0 <= rsi_14[i] <= 65.0
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG ENTRY: 4h uptrend + RSI pullback + session + volume
        if price_above_hma_4h and rsi_long_pullback and in_session and vol_confirms:
            # Add funding confirmation (optional boost)
            if funding_bullish or not funding_bearish:
                # Macro bias check
                if price_above_hma_1d or not price_below_hma_1d:
                    new_signal = POSITION_SIZE
        
        # SHORT ENTRY: 4h downtrend + RSI pullback + session + volume
        elif price_below_hma_4h and rsi_short_pullback and in_session and vol_confirms:
            # Add funding confirmation (optional boost)
            if funding_bearish or not funding_bullish:
                # Macro bias check
                if price_below_hma_1d or not price_above_hma_1d:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # Keep position if still in trend and RSI not at extreme
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if 4h still bullish and RSI < 70
                if price_above_hma_4h and rsi_14[i] < 70.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if 4h still bearish and RSI > 30
                if price_below_hma_4h and rsi_14[i] > 30.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND CHANGE ===
        if in_position and position_side > 0:
            if price_below_hma_4h and price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_4h and price_above_hma_1d:
                new_signal = 0.0
        
        # === RSI EXTREME EXIT ===
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals