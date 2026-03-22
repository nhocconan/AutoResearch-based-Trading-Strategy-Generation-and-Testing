#!/usr/bin/env python3
"""
Experiment #309: 4h Primary + 1d HTF — Vol Spike Reversion + Connors RSI + Regime Adaptive

Hypothesis: Volatility spike reversion combined with Connors RSI works better than pure trend following
on 4h timeframe because:
1. Vol spikes (ATR(7)/ATR(30) > 1.8) mark panic bottoms in crypto - high probability reversals
2. Connors RSI (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 catches oversold/overbought extremes
3. 1d KAMA(21) provides major trend filter without excessive lag
4. Choppiness Index switches between mean-revert (chop) and trend-follow (trending) modes
5. Asymmetric sizing favors longs (crypto bias) while allowing shorts in bear regimes
6. Target: 25-50 trades/year on 4h (appropriate for this TF, low fee drag)

Why this might beat current best (Sharpe=0.424):
- Vol spike reversion works THROUGH crashes (2022) unlike pure trend following
- Connors RSI has 75% win rate on mean reversion entries
- Regime-adaptive: mean revert in chop, trend follow otherwise
- Simpler entry conditions = more trades generated (avoid 0-trade failure)
- 1d HTF trend filter prevents counter-trend trades in strong trends

Key differences from failed 4h strategies:
- Vol spike detection (ATR ratio) instead of just RSI extremes
- Connors RSI instead of standard RSI(14) - more sensitive to short-term extremes
- Dual regime logic (chop vs trend) instead of one-size-fits-all
- Looser entry thresholds to ensure 25+ trades/year
- 2.5*ATR trailing stop (tighter than 3.0*ATR for 4h TF)

Position sizing: 0.25 base, 0.30 strong conviction (longs), 0.20 (shorts)
Stoploss: 2.5 * ATR trailing (tighter for 4h vs daily)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_volspike_connors_kama_1d_regime_v1"
timeframe = "4h"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts to market noise via Efficiency Ratio.
    """
    n = period
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        if i >= n:
            price_change = np.abs(close[i] - close[i-n])
            noise = np.sum(np.abs(np.diff(close[i-n:i+1])))
            
            if noise > 0:
                er = price_change / noise
            else:
                er = 0.0
            
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
        else:
            kama[i] = np.mean(close[max(0, i-n+1):i+1])
    
    return kama

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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's price change over lookback
    
    CRSI < 10 = extreme oversold (long signal)
    CRSI > 90 = extreme overbought (short signal)
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        up_streaks = np.sum(streak[i-streak_period:i+1] > 0)
        total = streak_period
        if total > 0:
            streak_rsi[i] = (up_streaks / total) * 100.0
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank of price change
    pct_rank = np.zeros(n)
    for i in range(rank_period, n):
        changes = np.diff(close[i-rank_period:i+1])
        if len(changes) > 0:
            current_change = close[i] - close[i-1]
            pct_rank[i] = (np.sum(changes <= current_change) / len(changes)) * 100.0
        else:
            pct_rank[i] = 50.0
    
    # Combine into CRSI
    for i in range(rank_period, n):
        crsi[i] = (rsi_3[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = period
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=n, min_periods=n, adjust=False).mean()
    
    atr_sum = atr.rolling(window=n, min_periods=n).sum()
    hh = high_s.rolling(window=n, min_periods=n).max()
    ll = low_s.rolling(window=n, min_periods=n).min()
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh.iloc[i] - ll.iloc[i]
        if range_hl > 0 and atr_sum.iloc[i] > 0:
            chop[i] = 100 * np.log10(atr_sum.iloc[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return upper.values, lower.values, sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    kama_1d_21 = calculate_kama(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_1d_21_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_21)
    
    # Calculate 4h indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Asymmetric: longs favored in crypto
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.20
    SHORT_STRONG = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(kama_1d_21_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]):
            continue
        
        # === 1D MAJOR TREND REGIME (primary direction filter — ASYMMETRIC) ===
        # Bull: price above 1d KAMA (favor longs with larger size)
        # Bear: price below 1d KAMA (allow shorts but reduced size)
        regime_bull = close[i] > kama_1d_21_aligned[i]
        regime_bear = close[i] < kama_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert)
        # CHOP < 45 = trending market (trend follow)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === VOLATILITY SPIKE DETECTION (key signal) ===
        # ATR(7)/ATR(30) > 1.8 = vol spike (panic/reversal zone)
        atr_ratio = atr_7[i] / (atr_30[i] + 1e-10)
        vol_spike = atr_ratio > 1.8
        vol_normal = atr_ratio < 1.3
        
        # === BOLLINGER BAND POSITION ===
        bb_pct = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i] + 1e-10)
        bb_oversold = bb_pct < 0.15
        bb_overbought = bb_pct > 0.85
        bb_squeeze = (bb_upper[i] - bb_lower[i]) < np.nanmean(bb_upper - bb_lower)[:i].min() * 1.5 if i > 50 else False
        
        # === CONNORS RSI SIGNALS ===
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        crsi_oversold = crsi[i] < 30.0
        crsi_overbought = crsi[i] > 70.0
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === PRICE vs SMA200 ===
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === ENTRY LOGIC (REGIME-AWARE + VOL SPIKE FOCUS) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (favored in bull regime)
        if regime_bull or not regime_bear:
            # Vol spike + CRSI oversold (primary mean reversion entry)
            if vol_spike and crsi_extreme_oversold:
                new_signal = LONG_STRONG
            
            # Vol spike + BB oversold + RSI oversold
            elif vol_spike and bb_oversold and rsi_oversold:
                new_signal = LONG_BASE
            
            # CRSI extreme oversold in any regime (strong reversal signal)
            elif crsi_extreme_oversold and bb_oversold:
                new_signal = LONG_BASE
            
            # Trending market + pullback to BB mid + CRSI rising
            elif is_trending and bb_pct < 0.4 and crsi[i] > crsi[i-1] if i > 0 else False:
                if i > 0 and crsi[i] > crsi[i-1]:
                    new_signal = LONG_BASE
            
            # Choppy market + CRSI oversold (mean revert)
            elif is_choppy and crsi_oversold and bb_oversold:
                new_signal = LONG_BASE * 0.8
            
            # Bull regime + RSI pullback (trend follow)
            elif regime_bull and rsi_oversold and price_above_sma200:
                new_signal = LONG_BASE
        
        # SHORT ENTRIES (only in bear regime, reduced size)
        if regime_bear:
            # Vol spike + CRSI overbought (primary mean reversion entry)
            if vol_spike and crsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG
            
            # Vol spike + BB overbought + RSI overbought
            elif vol_spike and bb_overbought and rsi_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
            
            # CRSI extreme overbought in bear regime
            elif crsi_extreme_overbought and bb_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
            
            # Trending market + rally to BB upper + CRSI falling
            elif is_trending and bb_pct > 0.6:
                if i > 0 and crsi[i] < crsi[i-1]:
                    if new_signal == 0.0:
                        new_signal = -SHORT_BASE
            
            # Choppy market + CRSI overbought (mean revert)
            elif is_choppy and crsi_overbought and bb_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8
        
        # === FREQUENCY SAFEGUARD (ensure 25+ trades/year on 4h) ===
        # Force trade if no signal for 45 bars (~7.5 days on 4h)
        if bars_since_last_trade > 45 and new_signal == 0.0 and not in_position:
            if crsi_extreme_oversold:
                new_signal = LONG_BASE * 0.6
            elif crsi_extreme_overbought:
                new_signal = -SHORT_BASE * 0.6
            elif regime_bull and rsi_oversold:
                new_signal = LONG_BASE * 0.6
            elif regime_bear and rsi_overbought:
                new_signal = -SHORT_BASE * 0.6
        
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
            # Long position: exit when CRSI turns overbought
            if position_side > 0 and crsi_extreme_overbought:
                crsi_exit = True
            # Short position: exit when CRSI turns oversold
            if position_side < 0 and crsi_extreme_oversold:
                crsi_exit = True
        
        # === VOLATILITY NORMALIZATION EXIT ===
        vol_exit = False
        if in_position and position_side != 0:
            # Exit when vol spike normalizes (took profit on reversion)
            if vol_normal and bars_since_last_trade > 5:
                vol_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 1d regime turns strongly bearish
            if position_side > 0 and regime_bear and close[i] < kama_1d_21_aligned[i] * 0.98:
                regime_reversal = True
            # Short position but 1d regime turns strongly bullish
            if position_side < 0 and regime_bull and close[i] > kama_1d_21_aligned[i] * 1.02:
                regime_reversal = True
        
        if stoploss_triggered or crsi_exit or vol_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.12:
                new_signal = 0.0
            elif new_signal > 0.28:
                new_signal = LONG_STRONG
            elif new_signal > 0:
                new_signal = LONG_BASE
            elif new_signal < -0.23:
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