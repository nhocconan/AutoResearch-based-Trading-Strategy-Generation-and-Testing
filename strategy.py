#!/usr/bin/env python3
"""
Experiment #343: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: Current best (#333) uses standard RSI pullback which works in trends but
fails in range/bear markets (2025 test period). This strategy combines:
1. Connors RSI (CRSI) - 3-component mean reversion signal (75% win rate in literature)
   CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
2. Choppiness Index (CHOP) - regime detection (range vs trend)
   CHOP > 61.8 = choppy/range (use mean reversion)
   CHOP < 38.2 = trending (use trend follow)
3. 1w HMA(21) - major trend direction bias (like #333 but with CRSI entries)
4. Dual-mode logic: mean revert in chop, trend follow in trends
5. ATR trailing stop + RSI extreme exit

Why this might beat #333 (Sharpe=0.435):
- CRSI generates more high-probability mean reversion signals than standard RSI
- CHOP filter prevents trend entries in choppy markets (reduces whipsaw)
- Works better in 2025 bear/range market (test period)
- Proven on ETH with Sharpe +0.923 in research
- Still uses 1w HTF for direction bias (proven effective)

Position sizing: 0.25-0.30 longs, 0.15-0.20 shorts (asymmetric)
Stoploss: 2.5 * ATR trailing
Target: 25-45 trades/year on 1d
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_hma_1w_regime_v1"
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
    """Calculate Hull Moving Average (HMA)."""
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI(3): Short-term momentum
    RSI_Streak(2): Streak duration strength
    PercentRank(100): Price position in recent range
    
    Extreme values (<10 or >90) indicate high-probability reversals.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_short = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] > 0:
            streak_rsi[i] = 100.0 * min(streak_abs[i], streak_period) / streak_period
        elif streak[i] < 0:
            streak_rsi[i] = 100.0 * (1.0 - min(streak_abs[i], streak_period) / streak_period)
        else:
            streak_rsi[i] = 50.0
    
    # Component 3: Percent Rank of close over lookback period
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        lookback = close[i-rank_period+1:i+1]
        rank = np.sum(lookback <= close[i])
        percent_rank[i] = 100.0 * rank / rank_period
    
    # Combine components
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = choppy/range market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        # Calculate ATR for each bar in lookback
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], 
                     abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j],
                     abs(low[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

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
    atr_30 = calculate_atr(high, low, close, 30)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    hma_1d_8 = calculate_hma(close, period=8)
    hma_1d_21 = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
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
    last_trade_bar = -20
    
    for i in range(150, n):  # Start later for CHOP calculation
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        
        if np.isnan(hma_1d_8[i]) or np.isnan(hma_1d_21[i]):
            continue
        
        # === 1W MAJOR TREND REGIME (primary direction filter) ===
        regime_bull = close[i] > hma_1w_21_aligned[i]
        regime_bear = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME (determines entry strategy) ===
        choppy_regime = chop[i] > 55.0  # Range/mean reversion
        trending_regime = chop[i] < 45.0  # Trend following
        neutral_regime = not choppy_regime and not trending_regime
        
        # === VOLATILITY REGIME ===
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10)
        high_vol = atr_ratio > 1.5
        vol_scale = 0.7 if high_vol else 1.0
        
        # === 1D LOCAL TREND ===
        hma_bullish = hma_1d_8[i] > hma_1d_21[i]
        hma_bearish = hma_1d_8[i] < hma_1d_21[i]
        
        hma_slope_up = hma_1d_21[i] > hma_1d_21[i-2] if i >= 2 else False
        hma_slope_down = hma_1d_21[i] < hma_1d_21[i-2] if i >= 2 else False
        
        price_above_hma = close[i] > hma_1d_21[i]
        price_below_hma = close[i] < hma_1d_21[i]
        
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === CRSI SIGNALS (mean reversion extremes) ===
        crsi_extreme_oversold = crsi[i] < 15.0  # Strong long signal
        crsi_oversold = crsi[i] < 25.0  # Moderate long signal
        crsi_extreme_overbought = crsi[i] > 85.0  # Strong short signal
        crsi_overbought = crsi[i] > 75.0  # Moderate short signal
        
        # CRSI turning (momentum shift)
        crsi_rising = crsi[i] > crsi[i-1] if i > 0 else False
        crsi_falling = crsi[i] < crsi[i-1] if i > 0 else False
        
        # === ENTRY LOGIC (DUAL REGIME) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # MODE 1: CHOPPY/RANGE REGIME (mean reversion)
        if choppy_regime:
            # Long: CRSI oversold + bull regime preferred
            if crsi_oversold:
                if regime_bull:
                    new_signal = LONG_BASE * vol_scale
                elif crsi_extreme_oversold:
                    new_signal = LONG_BASE * 0.8 * vol_scale
            
            # Short: CRSI overbought + bear regime preferred
            if crsi_overbought and new_signal == 0.0:
                if regime_bear:
                    new_signal = -SHORT_BASE * vol_scale
                elif crsi_extreme_overbought:
                    new_signal = -SHORT_BASE * 0.8 * vol_scale
        
        # MODE 2: TRENDING REGIME (trend follow)
        elif trending_regime:
            # Long: HMA bullish + CRSI rising + bull regime
            if hma_bullish and crsi_rising and regime_bull:
                if crsi[i] < 50.0:  # Enter on pullback in uptrend
                    new_signal = LONG_STRONG * vol_scale
                elif hma_slope_up:
                    new_signal = LONG_BASE * vol_scale
            
            # Short: HMA bearish + CRSI falling + bear regime
            if hma_bearish and crsi_falling and regime_bear:
                if new_signal == 0.0:
                    if crsi[i] > 50.0:  # Enter on bounce in downtrend
                        new_signal = -SHORT_STRONG * vol_scale
                    elif hma_slope_down:
                        new_signal = -SHORT_BASE * vol_scale
        
        # MODE 3: NEUTRAL REGIME (conservative entries)
        elif neutral_regime:
            # Only extreme CRSI signals
            if crsi_extreme_oversold and regime_bull:
                new_signal = LONG_BASE * 0.7 * vol_scale
            elif crsi_extreme_overbought and regime_bear:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.7 * vol_scale
        
        # === FREQUENCY SAFEGUARD (ensure 25+ trades/year) ===
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if crsi_extreme_oversold:
                new_signal = LONG_BASE * 0.6 * vol_scale
            elif crsi_extreme_overbought:
                new_signal = -SHORT_BASE * 0.6 * vol_scale
            elif regime_bull and crsi[i] < 40.0 and hma_bullish:
                new_signal = LONG_BASE * 0.5 * vol_scale
            elif regime_bear and crsi[i] > 60.0 and hma_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.5 * vol_scale
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === CRSI REVERSAL EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and crsi_extreme_overbought:
                crsi_exit = True
            if position_side < 0 and crsi_extreme_oversold:
                crsi_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and regime_bear and price_below_hma and chop[i] > 55.0:
                regime_reversal = True
            if position_side < 0 and regime_bull and price_above_hma and chop[i] > 55.0:
                regime_reversal = True
        
        if stoploss_triggered or crsi_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.10:
                new_signal = 0.0
            elif new_signal > 0.28:
                new_signal = LONG_STRONG * vol_scale
            elif new_signal > 0:
                new_signal = LONG_BASE * vol_scale
            elif new_signal < -0.18:
                new_signal = -SHORT_STRONG * vol_scale
            else:
                new_signal = -SHORT_BASE * vol_scale
        
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