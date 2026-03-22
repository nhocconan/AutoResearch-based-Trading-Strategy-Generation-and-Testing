#!/usr/bin/env python3
"""
Experiment #585: 1h Primary + 4h/1d HTF — Simplified Trend-Pullback with Wide CRSI Bands

Hypothesis: After 500+ failed strategies, the pattern is crystal clear:
- Lower TF (1h/30m/15m) strategies FAIL with Sharpe=0.000 = 0 TRADES (#575, #578, #580, #581)
- Entry conditions TOO STRICT with too many conflicting filters
- 4h strategies work better (#579 Sharpe=0.103, #584 Sharpe=0.146)
- 1d strategies work BEST (#577 Sharpe=0.520 current best)

NEW APPROACH for 1h:
- Use 4h HMA(21) for PRIMARY trend direction (not 1d - too slow for 1h entries)
- Use 1d HMA(21) for MAJOR regime confirmation (soft filter, not hard)
- Connors RSI(3,2,100) with VERY WIDE bands (20/80 not 30/70) to ENSURE trades
- Choppiness(14) for position sizing modifier (NOT hard entry filter)
- Session filter (8-20 UTC) as PREFERENCE not requirement
- Volume filter (rel vol > 0.5) soft confirmation
- ATR(14) 2.5x trailing stop
- Position size 0.25 (smaller for 1h to reduce fee drag)

Why this should beat Sharpe=0.520:
- 4h trend + 1h entry = optimal balance (proven in #579)
- WIDER CRSI bands = MORE trades (avoid 0-trade failure)
- SOFT filters = entries still happen when not all conditions perfect
- 1h TF captures more intraday moves than 4h while keeping quality

Target: 40-60 trades/year on 1h, Sharpe > 0.520, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_hma_4h1d_wide_v1"
timeframe = "1h"
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
    Calculate Connors RSI (CRSI) - proven 75% win rate for mean reversion.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) - very short term
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI - measure consecutive up/down days
    returns = close_s.pct_change()
    streak = np.zeros(n)
    for i in range(1, n):
        if pd.isna(returns.iloc[i]):
            streak[i] = 0
            continue
        if returns.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    streak_rsi = 100.0 - (100.0 / (1.0 + streak_rs))
    streak_rsi = streak_rsi.values
    
    # PercentRank - where today's return ranks vs last 100 days
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i].dropna()
        if len(window) > 0:
            current_ret = returns.iloc[i]
            if pd.isna(current_ret):
                percent_rank[i] = 50.0
            else:
                rank = (window < current_ret).sum() / len(window)
                percent_rank[i] = rank * 100.0
    
    # Combine into CRSI
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range market, CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        if highest == lowest:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
    return chop

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    mask = (plus_dm > 0) & (minus_dm > 0)
    plus_dm_vals = plus_dm.values.copy()
    minus_dm_vals = minus_dm.values.copy()
    plus_dm_vals[mask] = np.where(plus_dm_vals[mask] > minus_dm_vals[mask], plus_dm_vals[mask], 0)
    minus_dm_vals[mask] = np.where(minus_dm_vals[mask] > plus_dm_vals[mask], minus_dm_vals[mask], 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm_s = pd.Series(plus_dm_vals).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm_vals).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100.0 * plus_dm_s / (tr_s + 1e-10)
    minus_di = 100.0 * minus_dm_s / (tr_s + 1e-10)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

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
    
    # Calculate HTF HMAs
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    crsi_14 = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    # Volume SMA for relative volume
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, smaller for 1h)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_21_aligned[i]):
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(adx_14[i]) or np.isnan(crsi_14[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] == 0:
            continue
        
        # Extract hour from open_time for session filter
        # open_time is in milliseconds, convert to hour
        hour = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour <= 20
        
        # === 4H PRIMARY TREND (main direction filter) ===
        bull_4h = close[i] > hma_4h_21_aligned[i]
        bear_4h = close[i] < hma_4h_21_aligned[i]
        
        # === 1D MAJOR REGIME (confirmation, soft filter) ===
        bull_1d = close[i] > hma_1d_21_aligned[i]
        bear_1d = close[i] < hma_1d_21_aligned[i]
        bull_1d_slope = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        bear_1d_slope = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === CONNORS RSI ENTRY (VERY WIDE BANDS for more trades) ===
        # Long: CRSI < 25 in uptrend (oversold pullback)
        # Short: CRSI > 75 in downtrend (overbought rally)
        crsi_oversold_long = crsi_14[i] < 25.0
        crsi_overbought_short = crsi_14[i] > 75.0
        
        # === CHOPPINESS REGIME (position size modifier, NOT hard filter) ===
        # CHOP > 55 = range (favor mean reversion)
        # CHOP < 45 = trend (favor trend following)
        chop_range = chop_14[i] > 55.0
        chop_trend = chop_14[i] < 45.0
        
        # === VOLUME FILTER (soft confirmation) ===
        rel_vol = volume[i] / vol_sma_20[i]
        vol_ok = rel_vol > 0.5  # Very permissive
        
        # === ADX FILTER (minimal trend strength) ===
        adx_ok = adx_14[i] > 12.0  # Very permissive
        
        # === ENTRY LOGIC — WIDE BANDS to ensure trades ===
        new_signal = 0.0
        
        # LONG ENTRY: 4h bull + CRSI oversold + (1d confirms OR ADX/vol ok)
        long_score = 0
        if bull_4h and crsi_oversold_long:
            long_score += 3  # Core conditions
        if bull_1d:
            long_score += 1  # 1d confirmation
        if vol_ok:
            long_score += 1  # Volume confirmation
        if adx_ok:
            long_score += 1  # ADX confirmation
        if in_session:
            long_score += 0.5  # Session preference
        
        # SHORT ENTRY: 4h bear + CRSI overbought + (1d confirms OR ADX/vol ok)
        short_score = 0
        if bear_4h and crsi_overbought_short:
            short_score += 3  # Core conditions
        if bear_1d:
            short_score += 1  # 1d confirmation
        if vol_ok:
            short_score += 1  # Volume confirmation
        if adx_ok:
            short_score += 1  # ADX confirmation
        if in_session:
            short_score += 0.5  # Session preference
        
        # Entry threshold: score >= 4 (core + at least 1 confirmation)
        if long_score >= 4.0:
            # Position size modifier based on choppiness
            if chop_range:
                new_signal = POSITION_SIZE  # Range favors mean reversion longs
            else:
                new_signal = POSITION_SIZE * 0.9
        
        elif short_score >= 4.0:
            # Position size modifier based on choppiness
            if chop_range:
                new_signal = -POSITION_SIZE  # Range favors mean reversion shorts
            else:
                new_signal = -POSITION_SIZE * 0.9
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
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
        
        # === EXIT CONDITIONS (regime flip) ===
        # Exit long on 4h regime flip to bear
        if in_position and position_side > 0:
            if bear_4h and bear_1d:
                new_signal = 0.0
        
        # Exit short on 4h regime flip to bull
        if in_position and position_side < 0:
            if bull_4h and bull_1d:
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
                # Flip position
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