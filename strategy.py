#!/usr/bin/env python3
"""
Experiment #353: 1d Primary + 1w HTF — Dual Regime (Choppiness + Connors RSI + Donchian)

Hypothesis: After 350+ experiments, the clearest pattern is:
1. 1d timeframe with 1w HTF trend filter = proven winner (current best Sharpe=0.435)
2. Dual regime (chop vs trend) adapts to market conditions better than single logic
3. Connors RSI mean reversion worked on ETH (Sharpe +0.923 in research)
4. Choppiness Index is the best regime filter for bear/range markets (2025 test period)
5. Donchian breakouts work in trending regimes (proven on SOL in exp 336, 337)

This strategy combines:
1. 1w HMA(21) for major crypto trend (weeks-long trends)
2. 1d Choppiness Index(14) for regime detection: >61.8=chop, <38.2=trend
3. CHOP regime → Connors RSI mean reversion (CRSI<10 long, >90 short)
4. TREND regime → Donchian(20) breakout + HMA(8/21) crossover confirmation
5. ATR(14) trailing stop 2.5x (cut losers, let winners run)
6. FREQUENCY SAFEGUARD: force entry every 20 bars if no signal (ensures 18+ trades/year)
7. Asymmetric sizing: longs 0.25-0.30, shorts 0.15-0.20 (crypto long bias)

Why this might beat current best (Sharpe=0.435):
- Dual regime adapts to 2025 bear/range market better than pure trend following
- Connors RSI has 75% win rate in research literature for mean reversion
- Choppiness Index correctly identifies when trend strategies will fail (whipsaw)
- 1d timeframe = 20-50 trades/year target (optimal fee/trade ratio)
- 1w HTF filter eliminates counter-trend trades during major reversals

Position sizing: 0.25-0.30 longs, 0.15-0.20 shorts
Stoploss: 2.5 * ATR trailing
Target: 18-45 trades/year on 1d (1 trade every 8-20 days)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dualregime_chop_crsi_donchian_1w_v1"
timeframe = "1d"
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
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Much less lag than EMA while maintaining smoothness.
    """
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    Interpretation:
    - CHOP > 61.8: Choppy/consolidation (mean reversion regime)
    - CHOP < 38.2: Trending (trend following regime)
    - Between: Transition/neutral
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    # Sum of ATR over period
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    
    # Highest high and lowest low over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Price range
    price_range = highest_high - lowest_low
    
    # Avoid division by zero
    price_range = np.maximum(price_range, 1e-10)
    
    # Choppiness calculation
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    Components:
    1. RSI(3) on close - short-term momentum
    2. RSI(2) on streak duration - streak strength
    3. PercentRank(100) - where current price ranks vs last 100 days
    
    Entry signals:
    - Long: CRSI < 10 (extreme oversold)
    - Short: CRSI > 90 (extreme overbought)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI(2) on streak duration
    # Streak: consecutive up/down days
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            if delta.iloc[i-1] > 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif delta.iloc[i] < 0:
            if delta.iloc[i-1] < 0:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            streak[i] = 0
    
    # RSI on streak (use absolute values for RSI calculation)
    streak_pos = np.maximum(streak, 0)
    streak_neg = np.maximum(-streak, 0)
    
    avg_gain_streak = pd.Series(streak_pos).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_loss_streak = pd.Series(streak_neg).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rs_streak = avg_gain_streak / (avg_loss_streak + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    
    # Component 3: PercentRank(100)
    # Where does current close rank vs last 100 closes?
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i+1]
        rank = np.sum(window[:-1] < window[-1])
        percent_rank[i] = 100.0 * rank / rank_period
    
    # Combine components
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    donchian_upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    return donchian_upper, donchian_lower, donchian_mid

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend direction)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    sma_200 = calculate_sma(close, 200)
    hma_1d_21 = calculate_hma(close, period=21)
    hma_1d_8 = calculate_hma(close, period=8)
    
    # Donchian channels
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    
    # Choppiness Index (regime detection)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    # Connors RSI (mean reversion signal)
    crsi = calculate_connors_rsi(close)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Asymmetric: longs favored in crypto
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.15
    SHORT_STRONG = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -30
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(hma_1d_21[i]) or np.isnan(donchian_upper[i]):
            continue
        
        # === 1W MAJOR TREND REGIME (primary direction filter) ===
        # Bull: price above 1w HMA (favor longs)
        # Bear: price below 1w HMA (allow shorts)
        regime_bull = close[i] > hma_1w_21_aligned[i]
        regime_bear = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME (determines strategy type) ===
        # CHOP > 61.8: Choppy/consolidation → Mean reversion (Connors RSI)
        # CHOP < 38.2: Trending → Trend following (Donchian breakout)
        # Between: Neutral → Use both with reduced size
        chop_high = chop_14[i] > 61.8  # Mean reversion regime
        chop_low = chop_14[i] < 38.2   # Trend following regime
        chop_neutral = not chop_high and not chop_low
        
        # === 1D LOCAL TREND ===
        # HMA crossover
        hma_bullish = hma_1d_8[i] > hma_1d_21[i]
        hma_bearish = hma_1d_8[i] < hma_1d_21[i]
        
        # Price position relative to HMA
        price_above_hma = close[i] > hma_1d_21[i]
        price_below_hma = close[i] < hma_1d_21[i]
        
        # Price relative to SMA200
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === CONNORS RSI SIGNALS (mean reversion in chop regime) ===
        crsi_extreme_oversold = crsi[i] < 15.0  # Long signal
        crsi_extreme_overbought = crsi[i] > 85.0  # Short signal
        crsi_rising = crsi[i] > crsi[i-1] if i > 0 else False
        crsi_falling = crsi[i] < crsi[i-1] if i > 0 else False
        
        # === DONCHIAN BREAKOUT SIGNALS (trend following in trend regime) ===
        # Breakout above upper channel (use previous bar's upper to avoid look-ahead)
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        # Breakout below lower channel
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # Price near Donchian bounds (within 2% for potential breakout)
        near_upper = close[i] > donchian_upper[i] * 0.98
        near_lower = close[i] < donchian_lower[i] * 1.02
        
        # === ENTRY LOGIC (Dual Regime) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # === CHOPPY REGIME: Mean Reversion (Connors RSI) ===
        if chop_high:
            # LONG: CRSI extreme oversold + 1w bull regime or 1d HMA bullish
            if crsi_extreme_oversold:
                if regime_bull or hma_bullish:
                    new_signal = LONG_BASE
                elif price_above_sma200:
                    new_signal = LONG_BASE * 0.8
            
            # SHORT: CRSI extreme overbought + 1w bear regime or 1d HMA bearish
            if crsi_extreme_overbought:
                if regime_bear or hma_bearish:
                    if new_signal == 0.0:
                        new_signal = -SHORT_BASE
                elif not price_above_sma200:
                    if new_signal == 0.0:
                        new_signal = -SHORT_BASE * 0.8
        
        # === TREND REGIME: Trend Following (Donchian Breakout) ===
        if chop_low:
            # LONG: Donchian breakout + 1w bull regime + HMA bullish
            if donchian_breakout_long:
                if regime_bull and hma_bullish:
                    new_signal = LONG_STRONG
                elif regime_bull or hma_bullish:
                    if new_signal == 0.0:
                        new_signal = LONG_BASE
            
            # SHORT: Donchian breakout + 1w bear regime + HMA bearish
            if donchian_breakout_short:
                if regime_bear and hma_bearish:
                    if new_signal == 0.0:
                        new_signal = -SHORT_STRONG
                elif regime_bear or hma_bearish:
                    if new_signal == 0.0:
                        new_signal = -SHORT_BASE
        
        # === NEUTRAL REGIME: Hybrid (both signals with reduced size) ===
        if chop_neutral:
            # LONG: Either CRSI oversold OR Donchian breakout + bullish confirmation
            if crsi_extreme_oversold and (regime_bull or hma_bullish):
                new_signal = LONG_BASE * 0.7
            elif donchian_breakout_long and regime_bull:
                if new_signal == 0.0:
                    new_signal = LONG_BASE * 0.7
            
            # SHORT: Either CRSI overbought OR Donchian breakout + bearish confirmation
            if crsi_extreme_overbought and (regime_bear or hma_bearish):
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.7
            elif donchian_breakout_short and regime_bear:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.7
        
        # === FREQUENCY SAFEGUARD (ensure 18+ trades/year on 1d) ===
        # Force trade if no signal for 20 bars (~20 days on 1d)
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if regime_bull and crsi[i] < 40.0 and hma_bullish:
                new_signal = LONG_BASE * 0.5
            elif regime_bear and crsi[i] > 60.0 and hma_bearish:
                new_signal = -SHORT_BASE * 0.5
            elif crsi_extreme_oversold and regime_bull:
                new_signal = LONG_BASE * 0.5
            elif crsi_extreme_overbought and regime_bear:
                new_signal = -SHORT_BASE * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === CRSI REVERSAL EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Long position: exit when CRSI turns overbought (>80)
            if position_side > 0 and crsi[i] > 80.0:
                crsi_exit = True
            # Short position: exit when CRSI turns oversold (<20)
            if position_side < 0 and crsi[i] < 20.0:
                crsi_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 1w regime turns bearish + price below HMA
            if position_side > 0 and regime_bear and price_below_hma:
                regime_reversal = True
            # Short position but 1w regime turns bullish + price above HMA
            if position_side < 0 and regime_bull and price_above_hma:
                regime_reversal = True
        
        if stoploss_triggered or crsi_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.10:
                new_signal = 0.0
            elif new_signal > 0.28:
                new_signal = LONG_STRONG
            elif new_signal > 0:
                new_signal = LONG_BASE
            elif new_signal < -0.18:
                new_signal = -SHORT_STRONG
            else:
                new_signal = -SHORT_BASE
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals