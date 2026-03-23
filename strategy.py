#!/usr/bin/env python3
"""
Experiment #008: 30m Primary + 4h/1d HTF — Connors RSI + HTF Trend + Volume + Session

Hypothesis: After 7 failed strategies, the pattern shows:
1. Complex regime filters (CHOP+ADX) may be overfiltering → 0 trades or poor timing
2. Connors RSI has proven 75% win rate in literature for mean-reversion
3. 30m TF needs VERY strict filters to avoid fee drag (target 30-80 trades/year)
4. Funding rate mean-reversion is BEST EDGE for BTC/ETH but requires funding data

This strategy uses:
- Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- 4h HMA(21) for trend direction bias
- 1d SMA(200) for long-term trend filter
- Volume confirmation (>0.8x 20-bar avg)
- Session filter (8-20 UTC only) — reduces noise during low-liquidity hours
- ATR(14) trailing stoploss at 2.5x

Why this might work:
- CRSI is specifically designed for short-term mean-reversion (2-5 day holds)
- 4h HMA provides trend bias without overfiltering (unlike 1d HMA)
- Session filter eliminates Asian session noise (major source of whipsaws)
- Conservative size (0.20) protects against 2022-style crashes
- Entry: CRSI<10 (long) or CRSI>90 (short) + HTF trend alignment

Position sizing: 0.20 (conservative for 30m per Rule 4 & trade frequency rules)
Target: 40-80 trades/year on 30m (with session filter + strict CRSI thresholds)
Stoploss: 2.5*ATR trailing

CRITICAL: Call get_htf_data() ONCE before loop, use aligned arrays inside.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_hma4h_sma1d_session_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of price change over lookback period
    
    Entry signals:
    - Long: CRSI < 10 (oversold)
    - Short: CRSI > 90 (overbought)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # Component 2: RSI of Streak (consecutive up/down days)
    delta = close_s.diff()
    streak = np.zeros(n)
    
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if delta.iloc[i-1] > 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if delta.iloc[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (normalize to 0-100)
    # Positive streak = bullish, negative = bearish
    streak_abs = np.abs(streak)
    streak_sign = np.sign(streak)
    
    # Create streak RSI: high positive streak = high RSI, high negative = low RSI
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        lookback = streak[max(0, i-streak_period):i+1]
        if len(lookback) > 0:
            avg_streak = np.mean(lookback)
            # Map to 0-100 range
            streak_rsi[i] = 50.0 + 50.0 * np.tanh(avg_streak / 3.0)
    
    # Component 3: Percent Rank of price change over 100 periods
    pct_rank = np.zeros(n)
    for i in range(rank_period, n):
        changes = close_s.iloc[i-rank_period:i].diff().dropna()
        if len(changes) > 0:
            current_change = close_s.iloc[i] - close_s.iloc[i-1]
            pct_rank[i] = 100.0 * (changes < current_change).sum() / len(changes)
    
    # Combine into CRSI
    crsi = (rsi_short + streak_rsi + pct_rank) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    return sma.values

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for trend direction
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d SMA(200) for long-term trend filter
    sma_1d = calculate_sma(df_1d['close'].values, period=200)
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 30m SMA(50) for intermediate trend
    sma_50 = calculate_sma(close, period=50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, conservative for 30m)
    POSITION_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    # Track ADX regime to prevent rapid flipping
    prev_trend_bias = 0  # 0=neutral, 1=bull, -1=bear
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(vol_sma[i]) or np.isnan(sma_50[i]):
            continue
        if np.isnan(sma_1d_aligned[i]):
            continue
        if atr_14[i] == 0 or vol_sma[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === 4H TREND BIAS ===
        hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-5] if i >= 5 else False
        hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-5] if i >= 5 else False
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 1D LONG-TERM TREND ===
        price_above_sma_1d = close[i] > sma_1d_aligned[i]
        price_below_sma_1d = close[i] < sma_1d_aligned[i]
        
        # === 30M INTERMEDIATE TREND ===
        price_above_sma_50 = close[i] > sma_50[i]
        price_below_sma_50 = close[i] < sma_50[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 10.0
        crsi_overbought = crsi[i] > 90.0
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_sma[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Only trade during session hours (reduces noise)
        if in_session and volume_confirmed:
            # --- LONG ENTRY ---
            # CRSI oversold + 4h trend not bearish + price above 1d SMA (bull market bias)
            if crsi_oversold:
                if price_above_sma_1d and (price_above_hma_4h or not hma_4h_slope_bear):
                    new_signal = POSITION_SIZE
            
            # --- SHORT ENTRY ---
            # CRSI overbought + 4h trend not bullish + price below 1d SMA (bear market bias)
            elif crsi_overbought:
                if price_below_sma_1d and (price_below_hma_4h or not hma_4h_slope_bull):
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # If already in position, hold unless stoploss or trend flip
        if in_position and new_signal == 0.0:
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
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if price_below_sma_1d and hma_4h_slope_bear:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_sma_1d and hma_4h_slope_bull:
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