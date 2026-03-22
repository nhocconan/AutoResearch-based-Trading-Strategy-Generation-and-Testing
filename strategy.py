#!/usr/bin/env python3
"""
Experiment #562: 12h Primary + 1d/1w HTF — Dual Regime (Choppiness + Connors RSI)

Hypothesis: Based on #557 (Sharpe=0.152 with dual regime chop + Connors on 1d),
a regime-switching strategy on 12h can work if entry conditions are not too strict.

Key insights from failed experiments:
- #550, #552, #558: 0 trades = entry conditions TOO STRICT (too many filters)
- #557: Positive Sharpe with Choppiness + Connors = this combo has merit
- 12h TF should generate 20-50 trades/year (Rule 10)

Strategy logic:
1. 1d HMA(21) for MAJOR trend bias (HTF direction)
2. Choppiness Index(14) for REGIME detection:
   - CHOP > 61.8 = range regime (mean reversion)
   - CHOP < 38.2 = trend regime (trend follow)
   - Between = neutral (reduce size or stay flat)
3. Entry signals by regime:
   - RANGE: Connors RSI < 15 = long, CRSI > 85 = short (mean reversion)
   - TREND: RSI(14) pullback in 1d HMA direction (trend follow)
4. ATR(14) 2.5x trailing stoploss
5. Position size: 0.30 discrete (smaller in neutral regime)

Why this might beat Sharpe=0.435:
- Dual regime adapts to market conditions (range vs trend)
- Connors RSI proven for mean reversion (75% win rate in literature)
- 12h TF = fewer trades = less fee drag than 1h/4h
- 1d HTF filter prevents major counter-trend losses
- SIMPLE entry conditions = more trades (avoid 0-trade failure)

Position sizing: 0.30 base, 0.15 in neutral regime
Stoploss: 2.5 * ATR trailing
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_chop_crsi_1d_v1"
timeframe = "12h"
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
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8 = choppy/range market (mean reversion works)
    - CHOP < 38.2 = trending market (trend following works)
    """
    n = len(close)
    chop = np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        atr_sum = np.sum(atr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high > lowest_low:
            chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral if no range
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return over last 100 days
    
    Entry signals:
    - CRSI < 10-15 = oversold (long)
    - CRSI > 85-90 = overbought (short)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) - very short term
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI of streak (using absolute streak values for RSI calc)
    streak_abs = np.abs(streak)
    streak_s = pd.Series(streak_abs)
    delta_streak = streak_s.diff()
    gain_streak = delta_streak.where(delta_streak > 0, 0.0)
    loss_streak = -delta_streak.where(delta_streak < 0, 0.0)
    avg_gain_streak = gain_streak.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_loss_streak = loss_streak.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    rs_streak = avg_gain_streak / (avg_loss_streak + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    
    # For direction: positive streak = bullish, negative = bearish
    streak_direction = np.sign(streak)
    rsi_streak_signed = rsi_streak.values * streak_direction
    
    # Percent Rank of returns
    returns = close_s.pct_change()
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period+1:i+1].dropna()
        if len(window) > 0:
            percent_rank[i] = (returns.iloc[:i+1].dropna() <= returns.iloc[i]).sum() / len(window) * 100.0
    
    # Combine into CRSI
    crsi = (rsi_short + rsi_streak_signed + percent_rank) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF HMA for major trend direction
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_connors_rsi(close)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE_TREND = 0.30  # Full size in clear regime
    POSITION_SIZE_NEUTRAL = 0.15  # Half size in neutral regime
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]) or np.isnan(crsi[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime_1d = close[i] > hma_1d_21_aligned[i]
        bear_regime_1d = close[i] < hma_1d_21_aligned[i]
        
        # 1d HMA slope for trend strength
        hma_1d_slope_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_1d_slope_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === CHOPPINESS INDEX REGIME DETECTION ===
        chop_value = chop_14[i]
        
        # Regime classification
        range_regime = chop_value > 61.8  # Mean reversion works
        trend_regime = chop_value < 38.2  # Trend following works
        neutral_regime = not range_regime and not trend_regime
        
        # === ENTRY LOGIC BY REGIME ===
        new_signal = 0.0
        current_size = POSITION_SIZE_NEUTRAL if neutral_regime else POSITION_SIZE_TREND
        
        # RANGE REGIME: Mean reversion with Connors RSI
        if range_regime:
            # Long: CRSI oversold + price above 1d HMA (bias long in uptrend)
            if crsi[i] < 15.0 and bull_regime_1d:
                new_signal = current_size
            # Short: CRSI overbought + price below 1d HMA (bias short in downtrend)
            elif crsi[i] > 85.0 and bear_regime_1d:
                new_signal = -current_size
            # Weaker signals in neutral 1d trend
            elif crsi[i] < 10.0:
                new_signal = current_size * 0.7
            elif crsi[i] > 90.0:
                new_signal = -current_size * 0.7
        
        # TREND REGIME: Trend following with RSI pullback
        elif trend_regime:
            # Long: 1d bull + RSI pullback (40-55) + 1d HMA slope up
            if bull_regime_1d and hma_1d_slope_bull:
                if 40.0 <= rsi_14[i] <= 55.0:
                    new_signal = current_size
                elif rsi_14[i] < 40.0:  # Deeper pullback
                    new_signal = current_size * 1.2
            # Short: 1d bear + RSI rally (45-60) + 1d HMA slope down
            elif bear_regime_1d and hma_1d_slope_bear:
                if 45.0 <= rsi_14[i] <= 60.0:
                    new_signal = -current_size
                elif rsi_14[i] > 60.0:  # Stronger rally
                    new_signal = -current_size * 1.2
        
        # NEUTRAL REGIME: Only extreme CRSI signals
        elif neutral_regime:
            if crsi[i] < 8.0:
                new_signal = POSITION_SIZE_NEUTRAL
            elif crsi[i] > 92.0:
                new_signal = -POSITION_SIZE_NEUTRAL
        
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
        # Exit long on 1d regime flip to bear with slope confirmation
        if in_position and position_side > 0:
            if bear_regime_1d and hma_1d_slope_bear:
                new_signal = 0.0
        
        # Exit short on 1d regime flip to bull with slope confirmation
        if in_position and position_side < 0:
            if bull_regime_1d and hma_1d_slope_bull:
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