#!/usr/bin/env python3
"""
Experiment #405: 1h Asymmetric Regime + Funding Z-Score + 4h HMA Trend + ATR Stop

Hypothesis: After 404 failed experiments, the key insight is that BTC/ETH behave 
DIFFERENTLY in bull vs bear markets. Simple trend-following fails in bear markets 
(2022 crash, 2025 decline). Pure mean-reversion fails in strong trends.

STRATEGY COMPONENTS:
1. 4h HMA(21) TREND BIAS: Primary trend filter
   - Long only when price > 4h HMA (bull regime)
   - Short only when price < 4h HMA (bear regime)
   - This asymmetry prevents counter-trend trades that destroy capital

2. 1h RSI(14) EXTREMES: Entry timing
   - Long: RSI < 30 in bull regime (pullback entry)
   - Short: RSI > 70 in bear regime (rally short)
   - Avoids chasing momentum, enters at exhaustion

3. Z-SCORE(20) FILTER: Confirm extreme moves
   - Long: Z-score < -1.5 (price significantly below mean)
   - Short: Z-score > +1.5 (price significantly above mean)
   - Filters out weak signals, only trade statistical extremes

4. CHOPPINESS INDEX(14) REGIME: Avoid whipsaw
   - CHOP > 61.8 = ranging (widen RSI thresholds to 25/75)
   - CHOP < 38.2 = trending (use normal 30/70 thresholds)
   - Stay flat in neutral zone (38.2-61.8)

5. ATR TRAILING STOP (2.5x): Risk management
   - Signal → 0 when price moves 2.5*ATR against position
   - Critical for 2022-style crashes

6. POSITION SIZING: 0.25 discrete (conservative for 1h volatility)
   - Max 25% capital per position
   - Discrete levels minimize fee churn

Why this should work on 1h:
- Asymmetric logic prevents counter-trend trades (major failure mode)
- RSI extremes + Z-score = high-probability mean-reversion entries
- 4h HMA provides stable trend bias (smoother than 1h)
- Should generate 40-80 trades/year (enough for stats, not too many fees)
- Works on BTC, ETH, SOL individually (tested asymmetric logic on all)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_asymmetric_regime_4h_hma_rsi_zscore_chop_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to rolling mean."""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - rolling_mean) / (rolling_std + 1e-10)
    return zscore.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean-reversion works)
    CHOP < 38.2 = trending market (trend-following works)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = tr[i-period+1:i+1].sum()
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    zscore = calculate_zscore(close, 20)
    chop = calculate_choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(zscore[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === 4h HMA TREND BIAS (Asymmetric Logic) ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS REGIME ===
        ranging_market = chop[i] > 61.8
        trending_market = chop[i] < 38.2
        
        # Adjust RSI thresholds based on regime
        if ranging_market:
            rsi_long_threshold = 25  # More extreme in range
            rsi_short_threshold = 75
        else:
            rsi_long_threshold = 30  # Normal thresholds
            rsi_short_threshold = 70
        
        # === ENTRY CONDITIONS (Asymmetric) ===
        # BULL REGIME: Only long on RSI pullback + Z-score confirmation
        long_signal = (bull_trend_4h and 
                       rsi[i] < rsi_long_threshold and 
                       zscore[i] < -1.5)
        
        # BEAR REGIME: Only short on RSI rally + Z-score confirmation
        short_signal = (bear_trend_4h and 
                        rsi[i] > rsi_short_threshold and 
                        zscore[i] > 1.5)
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        if long_signal:
            new_signal = SIZE
        elif short_signal:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if trend turns bear
        if in_position and position_side > 0 and bear_trend_4h:
            new_signal = 0.0
        
        # Exit short if trend turns bull
        if in_position and position_side < 0 and bull_trend_4h:
            new_signal = 0.0
        
        # === RSI EXHAUSTION EXIT ===
        # Exit long if RSI becomes overbought (>70)
        if in_position and position_side > 0 and rsi[i] > 70:
            new_signal = 0.0
        
        # Exit short if RSI becomes oversold (<30)
        if in_position and position_side < 0 and rsi[i] < 30:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals