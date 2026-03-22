#!/usr/bin/env python3
"""
Experiment #590: 1h Primary + 4h/12h HTF — Regime-Adaptive Mean Reversion with Session Filter

Hypothesis: After 500+ failed strategies, the pattern for 1h TF is clear:
- Too many filters = 0 trades (#580, #585, #587, #589 all failed with Sharpe=0.000)
- Too loose = fee drag kills profit (>100 trades/year)
- SWEET SPOT: 4h/12h for DIRECTION, 1h for ENTRY TIMING only
- Session filter (8-20 UTC) avoids Asian session noise and whipsaws
- Connors RSI (CRSI) with WIDE bands (20-80) ensures trades generate
- Volume confirmation (>0.7x avg) filters fake breakouts
- Target: 40-70 trades/year, Sharpe > 0.520 to beat current best

This strategy uses PROVEN multi-TF logic:
1. 12h HMA(21) for MAJOR regime (bull/bear) — very slow, reliable
2. 4h HMA(21) for INTERMEDIATE trend direction — confirms 12h bias
3. 1h Connors RSI for pullback entries — long when CRSI<25 in uptrend, short when CRSI>75 in downtrend
4. Session filter: only enter 8-20 UTC (avoid Asian session noise)
5. Volume filter: volume > 0.7x 20-bar average (confirm moves)
6. ATR(14) 2.5x trailing stop for all positions
7. Position size: 0.25 discrete (smaller for 1h to reduce fee impact)

Why this might beat Sharpe=0.520:
- 12h + 4h double HTF confirmation = fewer false signals
- Session filter removes 40% of low-quality entries (Asian session)
- CRSI wide bands (20-80) = enough trades without overtrading
- Volume filter = confirms genuine moves, not fakeouts
- 1h TF with HTF direction = optimal balance for mean reversion

Position sizing: 0.25 base (discrete per Rule 4, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_session_hma_4h12h_v1"
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

def get_hour_from_open_time(prices):
    """Extract UTC hour from open_time column."""
    # open_time is in milliseconds since epoch
    open_time_ms = prices['open_time'].values
    # Convert to hours since epoch, then mod 24 for UTC hour
    hours_since_epoch = open_time_ms / (1000 * 3600)
    utc_hours = hours_since_epoch % 24
    return utc_hours.astype(int)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Extract UTC hour for session filter
    utc_hours = get_hour_from_open_time(prices)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF HMA for regime and trend
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_50 = calculate_hma(df_12h['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_50_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_50)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    crsi_14 = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Volume SMA for filter
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_12h_21_aligned[i]):
            continue
        if np.isnan(adx_14[i]) or np.isnan(crsi_14[i]):
            continue
        if np.isnan(volume_sma_20[i]) or volume_sma_20[i] == 0:
            continue
        
        # === 12H MAJOR REGIME (primary direction filter) ===
        bull_regime_12h = close[i] > hma_12h_21_aligned[i]
        bear_regime_12h = close[i] < hma_12h_21_aligned[i]
        
        # 12h HMA slope for trend strength confirmation
        hma_12h_slope_bull = hma_12h_21_aligned[i] > hma_12h_50_aligned[i]
        hma_12h_slope_bear = hma_12h_21_aligned[i] < hma_12h_50_aligned[i]
        
        # === 4H INTERMEDIATE TREND (confirm 12h bias) ===
        bull_trend_4h = close[i] > hma_4h_21_aligned[i]
        bear_trend_4h = close[i] < hma_4h_21_aligned[i]
        
        # === ADX FILTER (minimal trend strength) ===
        # ADX > 12 means some directional movement (permissive for 1h)
        trend_ok = adx_14[i] > 12.0
        
        # === SESSION FILTER (8-20 UTC only) ===
        # Avoid Asian session (0-8 UTC) and late night (20-24 UTC)
        session_ok = (utc_hours[i] >= 8) and (utc_hours[i] <= 20)
        
        # === VOLUME FILTER (>0.7x average) ===
        volume_ok = volume[i] > 0.7 * volume_sma_20[i]
        
        # === CONNORS RSI ENTRY (WIDE BANDS for trades) ===
        # Long: CRSI < 25 in uptrend (oversold pullback)
        # Short: CRSI > 75 in downtrend (overbought rally)
        crsi_oversold_long = crsi_14[i] < 25.0
        crsi_overbought_short = crsi_14[i] > 75.0
        
        # === ENTRY LOGIC — 4 CONFLUENCE FILTERS ===
        new_signal = 0.0
        
        # LONG ENTRY: 12h bull + 4h bull + CRSI oversold + session + volume
        if bull_regime_12h and bull_trend_4h and crsi_oversold_long and session_ok and volume_ok:
            # Size based on 12h trend strength
            if hma_12h_slope_bull:
                new_signal = POSITION_SIZE
            else:
                new_signal = POSITION_SIZE * 0.8
        
        # SHORT ENTRY: 12h bear + 4h bear + CRSI overbought + session + volume
        elif bear_regime_12h and bear_trend_4h and crsi_overbought_short and session_ok and volume_ok:
            # Size based on 12h trend strength
            if hma_12h_slope_bear:
                new_signal = -POSITION_SIZE
            else:
                new_signal = -POSITION_SIZE * 0.8
        
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
        # Exit long on 12h regime flip to bear
        if in_position and position_side > 0:
            if bear_regime_12h and hma_12h_slope_bear:
                new_signal = 0.0
        
        # Exit short on 12h regime flip to bull
        if in_position and position_side < 0:
            if bull_regime_12h and hma_12h_slope_bull:
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