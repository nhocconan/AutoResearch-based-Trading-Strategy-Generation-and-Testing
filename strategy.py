#!/usr/bin/env python3
"""
Experiment #377: 12h Multi-Timeframe Trend Alignment + RSI Pullback + Weekly Regime Filter

Hypothesis: After analyzing 376 experiments, the key insight is that successful strategies
need THREE timeframes aligned: Weekly (macro regime), Daily (intermediate trend), and
Primary (entry timing). Most failed strategies only used 2 timeframes or had conflicting signals.

STRATEGY COMPONENTS:
1. 1w HMA(21) = MACRO REGIME FILTER
   - Only long when price > 1w HMA (bullish macro)
   - Only short when price < 1w HMA (bearish macro)
   - This avoids counter-trend trades that destroyed capital in 2022 crash

2. 1d HMA(21) = INTERMEDIATE TREND BIAS
   - Confirms daily trend direction
   - Smoother than EMA, less whipsaw than SMA
   - Must align with weekly for signal generation

3. 12h ADX(14) > 22 = TREND STRENGTH CONFIRMATION
   - ADX > 22 = trending market (trend-following works)
   - ADX < 22 = ranging market (stay flat, avoid whipsaw)
   - Lower threshold than typical 25 to generate more signals on 12h

4. 12h RSI(14) PULLBACK = ENTRY TIMING
   - Long: RSI 35-50 in bullish regime (buy the dip)
   - Short: RSI 50-65 in bearish regime (sell the rip)
   - Better entry prices than breakout strategies

5. ATR(14) TRAILING STOP = RISK MANAGEMENT
   - Stoploss at 2.5 * ATR from entry
   - Trailing stop locks in profits
   - Signal → 0 when stopped out

6. POSITION SIZING = 0.30 DISCRETE
   - Conservative sizing (max 30% capital)
   - Discrete levels minimize fee churn
   - BTC 77% crash in 2022 → only -23% equity loss at this size

Why 12h timeframe:
- Slower than 4h → fewer signals but higher quality
- Less noise than 1h/4h strategies that failed
- Generates 20-40 trades/year (enough for stats, not too many for fees)
- Works with daily/weekly HTF data naturally

Why this should beat Sharpe=0.676:
- 3-timeframe alignment reduces false signals
- Weekly filter prevents disaster trades (2022 crash protection)
- RSI pullback entries have better risk/reward than breakouts
- ADX filter avoids choppy market whipsaw
- Conservative sizing keeps drawdown under control

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_trend_alignment_1d_1w_hma_rsi_pullback_adx_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    ADX > 25 = trending, ADX < 20 = ranging
    Uses Wilder's smoothing for +DI, -DI, and ADX
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    # Calculate True Range and Directional Movement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smooth with Wilder's method (equivalent to EMA with span=period)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate +DI and -DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Smooth DX to get ADX
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx[:] = adx_series.values
    
    return adx

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Default to neutral if undefined
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO REGIME FILTER (1w HMA) ===
        bull_macro = close[i] > hma_1w_aligned[i]
        bear_macro = close[i] < hma_1w_aligned[i]
        
        # === INTERMEDIATE TREND BIAS (1d HMA) ===
        bull_intermediate = close[i] > hma_1d_aligned[i]
        bear_intermediate = close[i] < hma_1d_aligned[i]
        
        # === TREND STRENGTH (ADX) ===
        trending = adx[i] > 22  # Lower threshold for 12h to get more signals
        
        # === ENTRY CONDITIONS ===
        # LONG: Bullish macro + Bullish intermediate + Trending + RSI pullback (35-50)
        long_condition = (
            bull_macro and 
            bull_intermediate and 
            trending and 
            35 <= rsi[i] <= 50
        )
        
        # SHORT: Bearish macro + Bearish intermediate + Trending + RSI pullback (50-65)
        short_condition = (
            bear_macro and 
            bear_intermediate and 
            trending and 
            50 <= rsi[i] <= 65
        )
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        if long_condition:
            new_signal = SIZE
        elif short_condition:
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
        # Exit long if macro or intermediate trend turns bearish
        if in_position and position_side > 0 and new_signal != 0.0:
            if bear_macro or bear_intermediate:
                new_signal = 0.0
        
        # Exit short if macro or intermediate trend turns bullish
        if in_position and position_side < 0 and new_signal != 0.0:
            if bull_macro or bull_intermediate:
                new_signal = 0.0
        
        # === ADX DROPOUT EXIT ===
        # Exit if trend strength collapses (ADX < 18 with hysteresis)
        if in_position and new_signal != 0.0:
            if adx[i] < 18:
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