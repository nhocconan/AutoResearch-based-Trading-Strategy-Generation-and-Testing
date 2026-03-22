#!/usr/bin/env python3
"""
Experiment #578: 30m Primary + 4h/1d HTF — Regime-Adaptive with Choppiness + CRSI

Hypothesis: After 500+ failed strategies, the pattern is clear:
- Simple trend following fails on BTC/ETH in bear/range markets (2022 crash, 2025 bear)
- Mean reversion alone fails in strong trends
- WINNING FORMULA: Regime detection (Choppiness) + Adaptive logic + HTF trend filter
- #577 (1d Chop+CRSI+1w) achieved Sharpe=0.520 — adapt this to 30m with 4h/1d HTF

Key innovations for 30m:
1. Choppiness Index (14) regime: CHOP>55 = range (mean revert), CHOP<45 = trend (follow)
2. Connors RSI (3,2,100) for precise entry timing — better than standard RSI
3. 4h HMA(21) for major trend direction, 1d HMA(50) for macro bias
4. Session filter (8-20 UTC) — avoid low liquidity Asian session
5. Volume filter (>0.8x 20-bar avg) — confirm moves with real participation
6. VERY STRICT entry: need 4+ confluence (regime + HTF + CRSI + session + volume)
7. Position size: 0.20 (smaller for 30m to handle more trades)
8. Target: 30-80 trades/year (NOT 200+ which kills profit via fees)

Why this might beat Sharpe=0.520:
- Regime adaptation works in BOTH bull and bear markets
- 4h HTF prevents major counter-trend losses
- CRSI catches reversals better than RSI (75% win rate in literature)
- Session/volume filters reduce false signals during low liquidity
- 30m entries within 4h trend = optimal frequency/quality balance

Position sizing: 0.20 base (discrete per Rule 4, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_crsi_regime_4h1d_v1"
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
    Calculate Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3): Short-term momentum
    2. RSI_Streak(2): RSI of consecutive up/down days
    3. PercentRank(100): Where current price ranks vs last 100 bars
    
    Entry signals:
    - Long: CRSI < 10 (extreme oversold)
    - Short: CRSI > 90 (extreme overbought)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streaks
    # Streak = consecutive up/down bars
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak values
    streak_s = pd.Series(streak)
    streak_gain = streak_s.diff().where(streak_s.diff() > 0, 0.0)
    streak_loss = -streak_s.diff().where(streak_s.diff() < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: PercentRank(100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        rank = np.sum(window < close[i])
        percent_rank[i] = rank / rank_period * 100.0
    
    # Combine into CRSI
    crsi = (rsi_3 + rsi_streak.values + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP)
    
    Formula: CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    Interpretation:
    - CHOP > 61.8: Range-bound market (mean reversion favored)
    - CHOP < 38.2: Trending market (trend following favored)
    - 38.2 - 61.8: Transition zone
    """
    n = len(close)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of ATR over period
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    price_range = highest_high - lowest_low
    price_range[price_range == 0] = 1e-10
    
    # CHOP formula
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    # Clip to valid range
    chop = np.clip(chop, 0, 100)
    
    return chop

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
    
    # Calculate 4h HTF HMA for major trend direction
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    
    # Calculate 1d HTF HMA for macro bias
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, 14)
    
    # Volume SMA for filter
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Smaller size for 30m (more trades potential)
    POSITION_SIZE = 0.20
    
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
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_50_aligned[i]):
            continue
        if np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        # Convert open_time to hour (open_time is in milliseconds)
        hour_utc = (open_time[i] // 1000 // 3600) % 24
        session_ok = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_sma_20[i]
        
        # === 4H MAJOR TREND (primary direction filter) ===
        bull_regime_4h = close[i] > hma_4h_21_aligned[i]
        bear_regime_4h = close[i] < hma_4h_21_aligned[i]
        
        # 4h HMA slope for trend strength
        hma_4h_slope_bull = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_slope_bear = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === 1D MACRO BIAS (higher TF confirmation) ===
        bull_macro_1d = close[i] > hma_1d_50_aligned[i]
        bear_macro_1d = close[i] < hma_1d_50_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range-bound (mean reversion favored)
        # CHOP < 45 = trending (trend following favored)
        range_regime = chop[i] > 55.0
        trend_regime = chop[i] < 45.0
        
        # === CONNORS RSI ENTRY SIGNALS ===
        # Long: CRSI < 15 (extreme oversold)
        # Short: CRSI > 85 (extreme overbought)
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === REGIME-ADAPTIVE ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY in RANGE REGIME (mean reversion)
        if range_regime and crsi_oversold and session_ok and volume_ok:
            # Need 4h not strongly bearish (avoid catching falling knife in strong downtrend)
            if not (bear_regime_4h and hma_4h_slope_bear):
                new_signal = POSITION_SIZE
        
        # LONG ENTRY in TREND REGIME (pullback in uptrend)
        elif trend_regime and bull_regime_4h and crsi_oversold and session_ok and volume_ok:
            if hma_4h_slope_bull:
                new_signal = POSITION_SIZE
            else:
                new_signal = POSITION_SIZE * 0.8
        
        # SHORT ENTRY in RANGE REGIME (mean reversion)
        elif range_regime and crsi_overbought and session_ok and volume_ok:
            # Need 4h not strongly bullish
            if not (bull_regime_4h and hma_4h_slope_bull):
                new_signal = -POSITION_SIZE
        
        # SHORT ENTRY in TREND REGIME (pullback in downtrend)
        elif trend_regime and bear_regime_4h and crsi_overbought and session_ok and volume_ok:
            if hma_4h_slope_bear:
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
        
        # === EXIT CONDITIONS (regime flip or CRSI reversal) ===
        # Exit long on 4h regime flip to strong bear
        if in_position and position_side > 0:
            if bear_regime_4h and hma_4h_slope_bear:
                new_signal = 0.0
            # Exit on CRSI overbought (mean reversion complete)
            elif crsi_overbought:
                new_signal = 0.0
        
        # Exit short on 4h regime flip to strong bull
        if in_position and position_side < 0:
            if bull_regime_4h and hma_4h_slope_bull:
                new_signal = 0.0
            # Exit on CRSI oversold (mean reversion complete)
            elif crsi_oversold:
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